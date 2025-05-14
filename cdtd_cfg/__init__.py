import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from .layers import MLP, CatEmbedding, Timewarp_Logistic, WeightNetwork
from .utils import (
    FastTensorDataLoader,
    LinearScheduler,
    cycle,
    low_discrepancy_sampler,
    set_seeds,
)


class MixedTypeDiffusion(nn.Module):
    # The core model within the CDTD
    def __init__(
        self,
        model, # takes in 'model', which is the MLP score function model
        dim, # All feature settings
        categories,
        proportions,
        num_features,
        sigma_data_cat, # noise hyperparams
        sigma_data_cont,
        sigma_min_cat,
        sigma_max_cat,
        sigma_min_cont,
        sigma_max_cont,
        cat_emb_init_sigma,
        timewarp_type="bytype", # tymewarp configs
        timewarp_weight_low_noise=1.0, # Initialized at 1.0, we do not weight low noise extra
    ):
        super(MixedTypeDiffusion, self).__init__() # Initializing all the functionality within the nn.Module

        self.dim = dim
        self.num_features = num_features
        self.num_cat_features = len(categories)
        self.num_cont_features = num_features - self.num_cat_features
        self.num_unique_cats = sum(categories)
        self.categories = categories
        self.model = model

        # learnable embedding for each cat value per feature (with optional bias per feature)
        self.cat_emb = CatEmbedding(dim, categories, cat_emb_init_sigma, bias=True) 
        
        # The variances of categorical and cont data entering the diff model
        # By using register buffer, they are added to the model state, but not trainable. Used in preconditioning formula later on.
        self.register_buffer("sigma_data_cat", torch.tensor(sigma_data_cat)) # So the intrinsic scales of the variables ( = 1 if normalized)
        self.register_buffer("sigma_data_cont", torch.tensor(sigma_data_cont))

        # For each categorical variable the entropy is computed. Used to normalize the losses to treat all features equally
        entropy = torch.tensor([-torch.sum(p * p.log()) for p in proportions])
        self.register_buffer(
            "normal_const",
            torch.cat((entropy, torch.ones((self.num_cont_features,)))), # cont variables = 1
        )
        # We create the weight network to predict the difficulty of different noise levels u, helping us reweigh the loss later 
        self.weight_network = WeightNetwork(1024) 
 
        # timewarping bounds are set, to bound the the noise used to normalize and denormalize sigma values in timewarping
        self.timewarp_type = timewarp_type
        self.sigma_min_cat = torch.tensor(sigma_min_cat) # So the min and max noise used for cat and cont variables are defined here
        self.sigma_max_cat = torch.tensor(sigma_max_cat)
        self.sigma_min_cont = torch.tensor(sigma_min_cont)
        self.sigma_max_cont = torch.tensor(sigma_max_cont)

        # combine sigma boundaries of cat and cont variables for transforming sigmas to [0,1]
        sigma_min = torch.cat(
            (
                torch.tensor(sigma_min_cat).repeat(self.num_cat_features),
                torch.tensor(sigma_min_cont).repeat(self.num_cont_features),
            ),
            dim=0,
        )
        sigma_max = torch.cat(
            (
                torch.tensor(sigma_max_cat).repeat(self.num_cat_features),
                torch.tensor(sigma_max_cont).repeat(self.num_cont_features),
            ),
            dim=0,
        )
        self.register_buffer("sigma_max", sigma_max) # These are again stored
        self.register_buffer("sigma_min", sigma_min)

        # Timewarping network is called using all the variables we just created and stored
        self.timewarp_cdf = Timewarp_Logistic(
            self.timewarp_type,
            self.num_cat_features,
            self.num_cont_features,
            sigma_min,
            sigma_max,
            weight_low_noise=timewarp_weight_low_noise,
            decay=0.0,
        )

    @property
    def device(self):
        return next(self.model.parameters()).device # Ensures all tensors are created on the same device?

    def diffusion_loss(self, x_cat_0, x_cont_0, cat_logits, cont_preds):
        assert len(cat_logits) == self.num_cat_features # checks shape of predicted values
        assert cont_preds.shape == x_cont_0.shape

        # cross entropy over categorical features for each individual 
        # cat_logits[i] is shaped: (B, C_i) so batch by number of categories for feature i.
        # x_cat_0 is shaped B and contains the true index (from 0 to C_i - 1)
        # for feature i the cross_entropy loss is computed between the expected logits and the true logits.
        ce_losses = torch.stack(
            [
                F.cross_entropy(cat_logits[i], x_cat_0[:, i], reduction="none")
                for i in range(self.num_cat_features)
            ],
            dim=1,
        ) # final shape of ce_losses = (B, num_cat_features)

        # MSE loss over numerical features (also returns B,num_cont_features)
        mse_losses = (cont_preds - x_cont_0) ** 2

        return ce_losses, mse_losses

    def add_noise(self, x_cat_emb_0, x_cont_0, sigma):
        # Adds gaussian noise to the inputs based on the sampled sigma values 
        sigma_cat = sigma[:, : self.num_cat_features]
        sigma_cont = sigma[:, self.num_cat_features :]

        # Cat features are increased using randn_like, which creates a tensor of the same shape, filled with N0,1. 
        # we multiply this tensor by the sigma (we use unsqueeze 2, to match the shapes, since the whole category embedding the same sigma)
        x_cat_emb_t = x_cat_emb_0 + torch.randn_like(x_cat_emb_0) * sigma_cat.unsqueeze(
            2
        ) 
        x_cont_t = x_cont_0 + torch.randn_like(x_cont_0) * sigma_cont # same, but without the need of unsqueezing

        return x_cat_emb_t, x_cont_t

    def loss_fn(self, x_cat, x_cont, u=None, cfg = False, y_condition_1=None, y_condition_2=None):
        batch = x_cat.shape[0] if x_cat is not None else x_cont.shape[0]

        # get ground truth data
        x_cat_emb_0 = self.cat_emb(x_cat) # Embedded the original categorical data
        x_cont_0 = x_cont # Storing the rest of the original training data
        x_cat_0 = x_cat

        # draw u and convert to standard deviations for noise
        with torch.no_grad():
            if u is None:
                u = low_discrepancy_sampler(batch, device=self.device)  # (B,), samples B values (0, 1) throug the sampler
            sigma = self.timewarp_cdf(u, invert=True).detach().to(torch.float32) # Creates the sigmas through the inverse timewarp cdf
            u = u.to(torch.float32) # u is transformed into a float 32
            # It is asserted that sigma has the correct shape (for every individual a sigma for each feature)
            assert sigma.shape == (batch, self.num_features) 

        x_cat_emb_t, x_cont_t = self.add_noise(x_cat_emb_0, x_cont_0, sigma) # Noise is added to the data
        # Model outputs based on noised data, u, sigma and the class labels
        cat_logits, cont_preds = self.precondition(x_cat_emb_t, x_cont_t, u, sigma, cfg = False, y_condition_1=None, y_condition_2=None) 
        ce_losses, mse_losses = self.diffusion_loss( 
            x_cat_0, x_cont_0, cat_logits, cont_preds 
        ) # The ce and mse losses are computed based on the predictions

        # compute EDM weight
        sigma_cont = sigma[:, self.num_cat_features :] # sigmas for the continuous features
        # EDM weight formula from the paper:
        # This is variance of continuous data + variance of noise added in the diffusion process/ (variance cont data * variance noise)^2
        # The smaller sigma_cont, the higher the weight
        cont_weight = (sigma_cont**2 + self.sigma_data_cont**2) / (
            (sigma_cont * self.sigma_data_cont) ** 2 + 1e-7
        )

        losses = {}
        losses["unweighted"] = torch.cat((ce_losses, mse_losses), dim=1) # all unweighted losses combined into a single tensor (B, num_feat)
        losses["unweighted_calibrated"] = losses["unweighted"] / self.normal_const # Categorical losses are calibrated using entropy
        weighted_calibrated = (     # MSE losses are scaled by EDM weights before combining them 
            torch.cat((ce_losses, cont_weight * mse_losses), dim=1) / self.normal_const
        )
        
        # Now the noise levels u are transformed using a log transformation for a better spread
        # Subsequently passed through the fourier transform with scalar output, which indicates the difficulty of denoising at u
        c_noise = torch.log(u.to(torch.float32) + 1e-8) * 0.25 
        time_reweight = self.weight_network(c_noise).unsqueeze(1) 
        
        # Timewarping loss: encourages the learned noise schedule (sigma = timewarp(u)) to match the actual average losses
        losses["timewarping"] = self.timewarp_cdf.loss_fn(
            sigma.detach(), losses["unweighted_calibrated"].detach()
        )

        # The weight network loss ensures predicted difficulty (time_reweight) matches the actual average loss across features        
        weightnet_loss = (
            time_reweight.exp() - weighted_calibrated.detach().mean(1)
        ) ** 2
        
        # Each feature is then reweighted by dividing the actual loss by the expected difficulty to focus on underperforming regions
        losses["weighted_calibrated"] = (
            weighted_calibrated / time_reweight.exp().detach()
        )

        # Final training loss combines reweighted loss, timewarp fitting loss, and weight network prediction loss        
        train_loss = (
            losses["weighted_calibrated"].mean() # How well does the model reconstruct the original data from its noised version?
            + losses["timewarping"].mean() # Given the models current ability, how should we warp the training noise distribution?
            + weightnet_loss.mean() # Is the models prediction of how hard u is actually accurate?
        )

        # final loss is stored
        losses["train_loss"] = train_loss

        return losses, sigma

    def precondition(self, x_cat_emb_t, x_cont_t, u, sigma, cfg = False, y_condition_1=None, y_condition_2=None):
        """
        Improved preconditioning proposed in the paper "Elucidating the Design
        Space of Diffusion-Based Generative Models" (EDM) adjusted for categorical data
        """
        # The idea is that the noised data should still be roughly similarly distributed despite the noise level
        # This means, that if we were to add a lot of noise, we need to rescale it to behave the same as slightly noised data

        # We retrieve the sigma levels of cat and cont variables
        sigma_cat = sigma[:, : self.num_cat_features]
        sigma_cont = sigma[:, self.num_cat_features :]

        # Scale each categorical embedding by a value that accounts for its noise, ensuring that the noisy inputs are scaled down
        c_in_cat = (
            1 / (self.sigma_data_cat**2 + sigma_cat.unsqueeze(2) ** 2).sqrt()
        )  # batch, num_features, 1
        # Same for cont (however this is not shaped with a third dimension)
        c_in_cont = 1 / (self.sigma_data_cont**2 + sigma_cont**2).sqrt()
        # c_noise = u.log() / 4, computing a time embedding for u
        c_noise = torch.log(u + 1e-8) * 0.25 * 1000   # * 1000 probably for model sensitivity 

        # Inputs are passed through the model, where the cat and cont embeddings are rescaled, and the noise level + labels are added
        cat_logits, cont_preds = self.model(
            c_in_cat * x_cat_emb_t,
            c_in_cont * x_cont_t,
            c_noise,
            cfg,
            y_condition_1,
            y_condition_2        
        )

        assert len(cat_logits) == self.num_cat_features
        assert cont_preds.shape == x_cont_t.shape

        # The model now combines the continuous prediction with the original noised data. This is done using c_skip and c_out
        # This is done using the sigma_cont (fixed variance of the features) and the sigma_data (the noise level in this sample)
        # The formulas are defined in a way that if the noise was small we rely more on the original x_cont_t data
        # If a lot of noise we rely more on the output. This preconditioning approach perfroms interpolation, making training more stable
        c_skip = self.sigma_data_cont**2 / (sigma_cont**2 + self.sigma_data_cont**2)
        c_out = (
            sigma_cont
            * self.sigma_data_cont
            / (sigma_cont**2 + self.sigma_data_cont**2).sqrt()
        )
        D_x = c_skip * x_cont_t + c_out * cont_preds
        
        # Importantly, this EDM method, instead of letting the model learn the x_0, it learns the direction of x_0.
        # Tihs is because we later interpolate it with the original data. This leads to better gradient flow and improved stability.
        return cat_logits, D_x

    def score_interpolation(self, x_cat_emb_t, cat_logits, sigma, return_probs=False):
        # Here we estimate the score function (gradient of log prob) for cat variables 
        
        if return_probs: # Indicates if the function should return probabilities
            # transform logits for categorical features to softmax probabilities
            probs = []
            for logits in cat_logits:
                probs.append(F.softmax(logits.to(torch.float64), dim=1))
            return probs # This output can be useful if we simply want to select the most likely category using argmax

        # If not, it returns the interpolated categories score
        def interpolate_emb(i): # Called for an individual feature
            p = F.softmax(cat_logits[i].to(torch.float64), dim=1) # Returns a probability that sample b belongs to category j
            true_emb = self.cat_emb.get_all_feat_emb(i).to(torch.float64) # Returns the true normalized embeddings of categories of feat i
            return torch.matmul(p, true_emb) # Returns a weighted average, so probability * category embedding, summed up.

        x_cat_emb_0_hat = torch.zeros_like( # an empty tensor created of the shape of x_cat_emb_t
            x_cat_emb_t, device=self.device, dtype=torch.float64
        )
        for i in range(self.num_cat_features):
            x_cat_emb_0_hat[:, i, :] = interpolate_emb(i) # For each categorical feature, it returns the interpolated embedding

        sigma_cat = sigma[:, : self.num_cat_features] # The variance of categorical variables is returned
        # The interpolated score is computed using the difference between clean and noised embedding, scaled by sigma
        interpolated_score = (x_cat_emb_t - x_cat_emb_0_hat) / sigma_cat.unsqueeze(2) 
        #The interpolated score shows what direction and how much (so how) to push the embedding in order to reduce noise
        return interpolated_score, x_cat_emb_0_hat # We return the interpolated score and interpolated embedding. 

    @torch.inference_mode()
    def sampler(self, cat_latents, cont_latents, num_steps=200, cfg = False, y_condition_1=None, y_condition_2=None, cfg_scale = 0.0):
        # The sampler is used to create synthetic samples using CDTD. It does so by running the reverse diffusion process.
        # This means it starts with pure noise, and removes it gradually (over num_steps)
        # The only other inputs are the categorical and continuous latent 
        B = ( # computes batch size
            cont_latents.shape[0]
            if self.num_cont_features > 0
            else cat_latents.shape[0]
        )

        # construct time steps [0,1] evenly spaced over n+1 points
        u_steps = torch.linspace(
            1, 0, num_steps + 1, device=self.device, dtype=torch.float64
        )
        t_steps = self.timewarp_cdf(u_steps, invert=True) # converts uniform variable u into actual noise levels at step t using inverse cdf

        # Makes sure that the first and last t_step approx match the min and max bounds for the noise schedule. 
        assert torch.allclose(t_steps[0].to(torch.float32), self.sigma_max.float())
        assert torch.allclose(t_steps[-1].to(torch.float32), self.sigma_min.float())

        # initialize latents at maximum noise level
        t_cat_next = t_steps[0, : self.num_cat_features] # Extract initial max noise level per feature type
        t_cont_next = t_steps[0, self.num_cat_features :]
        # Multiply initial noise level by latents. These are standard normal tensors. So this is like ε * σ_max. 
        x_cat_next = cat_latents.to(torch.float64) * t_cat_next.unsqueeze(1) # Shape: (B, num_cat_features, emb_dim)
        x_cont_next = cont_latents.to(torch.float64) * t_cont_next # Shape: (B, num_cont_features)

        # We loop through all denoising steps in reverse order (so max noise to min)
        for i, (t_cur, t_next, u_cur) in enumerate( 
            zip(t_steps[:-1], t_steps[1:], u_steps[:-1])
        ): # Fetch current sigma level, sigma after the denoising step, and the corresponding u value
            
            t_cur = t_cur.repeat((B, 1)) # The noise levels are repeated to match the batch dimensions
            t_next = t_next.repeat((B, 1)) 
            t_cont_cur = t_cur[:, self.num_cat_features :] 

            # The model is applied to the current noised data. 
            # The output is the current categorical class logits and denoised versions of cont features
            
            if cfg:
                cat_logits, x_cont_denoised = self.precondition(
                    x_cat_emb_t=x_cat_next.to(torch.float32),
                    x_cont_t=x_cont_next.to(torch.float32),
                    u=u_cur.to(torch.float32).repeat((B,)),
                    sigma=t_cur.to(torch.float32),
                    cfg = cfg,
                    y_condition_1=None,
                    y_condition_2=None
                )
                d_cat_unc, _ = self.score_interpolation(x_cat_next, cat_logits, t_cur) # categorical score vector
                d_cont_unc = (x_cont_next - x_cont_denoised.to(torch.float64)) / t_cont_cur # continuous score vector

                cat_logits_1, x_cont_denoised_1 = self.precondition(
                    x_cat_emb_t=x_cat_next.to(torch.float32),
                    x_cont_t=x_cont_next.to(torch.float32),
                    u=u_cur.to(torch.float32).repeat((B,)),
                    sigma=t_cur.to(torch.float32),
                    cfg = cfg,
                    y_condition_1=y_condition_1,
                    y_condition_2=None
                )
                d_cat_con_1, _ = self.score_interpolation(x_cat_next, cat_logits, t_cur) # categorical score vector
                d_cont_con_1 = (x_cont_next - x_cont_denoised.to(torch.float64)) / t_cont_cur # continuous score vector
                
                cat_logits_2, x_cont_denoised_2 = self.precondition(
                    x_cat_emb_t=x_cat_next.to(torch.float32),
                    x_cont_t=x_cont_next.to(torch.float32),
                    u=u_cur.to(torch.float32).repeat((B,)),
                    sigma=t_cur.to(torch.float32),
                    cfg = cfg,
                    y_condition_1=None,
                    y_condition_2=y_condition_2
                )
                d_cat_con_2, _ = self.score_interpolation(x_cat_next, cat_logits, t_cur) # categorical score vector
                d_cont_con_2 = (x_cont_next - x_cont_denoised.to(torch.float64)) / t_cont_cur # continuous score vector
                
                # Combining the categorical and continuous score vectors
                d_cat_cur = d_cat_unc + cfg_scale * (d_cat_con_1 * 0.5 + d_cat_con_2 * 0.5 - d_cat_unc) 
                d_cont_cur = d_cont_unc + cfg_scale * (d_cont_con_1 * 0.5 + d_cont_con_2 * 0.5 - d_cat_unc) 
                
            else:
                cat_logits, x_cont_denoised = self.precondition(
                    x_cat_emb_t=x_cat_next.to(torch.float32),
                    x_cont_t=x_cont_next.to(torch.float32),
                    u=u_cur.to(torch.float32).repeat((B,)),
                    sigma=t_cur.to(torch.float32),
                    cfg = cfg,
                    y_condition_1=y_condition_1,
                    y_condition_2=y_condition_2
                )

                # estimate scores
                d_cat_cur, _ = self.score_interpolation(x_cat_next, cat_logits, t_cur) # categorical score vector
                d_cont_cur = (x_cont_next - x_cont_denoised.to(torch.float64)) / t_cont_cur # continuous score vector
                
            # adjust data samples through euler updates. 
            h = t_next - t_cur # The difference between the current and next t is computed as a step size. (negative value)
            # We now remove the step size * score function to move towards the right direction in the next step
            x_cat_next = (
                x_cat_next + h[:, : self.num_cat_features].unsqueeze(2) * d_cat_cur
            )
            x_cont_next = x_cont_next + h[:, self.num_cat_features :] * d_cont_cur

        # final prediction of classes for categorical feature (so the prediction of the last denoising step)
        u_final = u_steps[:-1][-1]
        t_final = t_steps[:-1][-1].repeat(B, 1)

        cat_logits, _ = self.precondition( # The model is run one final time to get the final logits of the cat variables
            x_cat_emb_t=x_cat_next.to(torch.float32),
            x_cont_t=x_cont_next.to(torch.float32),
            u=u_final.to(torch.float32).repeat((B,)),
            sigma=t_final.to(torch.float32),
            cfg = cfg,
            y_condition_1=y_condition_1,
            y_condition_2=y_condition_2            
        )

        # get probabilities for each category and sample the most likely per feature
        probs = self.score_interpolation(
            x_cat_next, cat_logits, t_final, return_probs=True
        )
        x_cat_gen = torch.empty(B, self.num_cat_features, device=self.device)
        for i in range(self.num_cat_features):
            x_cat_gen[:, i] = probs[i].argmax(1)

        # final output is shaped (B, F_cat) (B, F_cont)
        return x_cat_gen.cpu(), x_cont_next.cpu() # .cpu() moves tensors from gpu to cpu memory.


class CDTD: # The all encompassing class we can call containing all training and sampling methods
    def __init__(
        self,
        X_cat_train,
        X_cont_train,
        cat_emb_dim=16, # Embedding dimensions of cat features
        mlp_emb_dim=256, # MLP architecture details (size of proj layer, timestep embedding that is added, and input of first mlp layer)
        mlp_n_layers=5,
        mlp_n_units=1024,
        sigma_data_cat=1.0, # st dev of the cont and cat data
        sigma_data_cont=1.0,
        sigma_min_cat=0.0, # Min and max sigma of cont and cat noise
        sigma_min_cont=0.0,
        sigma_max_cat=100.0,
        sigma_max_cont=80.0,
        cat_emb_init_sigma=0.001, # How strongly to initialize cat embeddings. So high means initial embeddings far apart, maybe instable
        timewarp_type="bytype",  # 'single', 'bytype', or 'all'
        timewarp_weight_low_noise=1.0, # Bias towards better learning low noise steps
        cfg = False, 
        y_condition_1=None, 
        y_condition_2=None
    ):
        super().__init__()

        self.num_cat_features = X_cat_train.shape[1]
        self.num_cont_features = X_cont_train.shape[1]
        self.num_features = self.num_cat_features + self.num_cont_features
        self.cat_emb_dim = cat_emb_dim

        # derive number of categories for each categorical feature
        self.categories = []
        for i in range(self.num_cat_features): # Number of unique values per feature counted and stored
            uniq_vals = np.unique(X_cat_train[:, i])
            self.categories.append(len(uniq_vals))

        # derive proportions for max CE losses at t = 1 for normalization
        self.proportions = []
        n_sample = X_cat_train.shape[0]
        for i in range(len(self.categories)): # for each categorical feature we compute the proportion of categories in the sample
            _, counts = X_cat_train[:, i].unique(return_counts=True)
            self.proportions.append(counts / n_sample)

        # Compute the number of distinct categories per label 
        if cfg:     
            num_classes_1 = len(np.unique(y_condition_1.numpy()))
            num_classes_2 = len(np.unique(y_condition_2.numpy()))
            print(f"[DEBUG] num_classes_1 (label_1 '{label_1}'): {num_classes_1}")
            print(f"[DEBUG] num_classes_2 (label_2 '{label_2}'): {num_classes_2}")

        score_model = MLP( # MLP is called to train the score NN. 
            self.num_cont_features,
            self.cat_emb_dim,
            self.categories,
            self.proportions, # Used in the MLP to set initialize cat bias
            mlp_emb_dim,
            mlp_n_layers,
            mlp_n_units,
            num_classes_1,
            num_classes_2
        )

        self.diff_model = MixedTypeDiffusion( # The mixed diffusion model is subsequently trained
            model=score_model,
            dim=self.cat_emb_dim,
            categories=self.categories,
            num_features=self.num_features,
            sigma_data_cat=sigma_data_cat,
            sigma_data_cont=sigma_data_cont,
            sigma_min_cat=sigma_min_cat,
            sigma_max_cat=sigma_max_cat,
            sigma_min_cont=sigma_min_cont,
            sigma_max_cont=sigma_max_cont,
            proportions=self.proportions, # Used to compute entropy, this entropy rescales the values.
            cat_emb_init_sigma=cat_emb_init_sigma, # Controls how cat embeddings are initialized
            timewarp_type=timewarp_type, 
            timewarp_weight_low_noise=timewarp_weight_low_noise,
        )

    def fit(  # Function that fits the CDTD
        self,
        X_cat_train,
        X_cont_train,
        num_steps_train=30_000, # total training steps
        num_steps_warmup=1000, # number of warm up steps
        batch_size=4096, 
        lr=1e-3, # learning rate, increasing leads to bigger jumps but more instable
        seed=42,
        ema_decay=0.999, # exponential moving average smoothing, decreasing means we follow last weights more closely, but more noisy
        log_steps=100, # print training steps every log_steps, so decreasing means more logs, might make it more noisy and less readable
        cfg = False, 
        y_condition_1=None, 
        y_condition_2=None,
        dropout_ratio = 1,
        condition_1_ratio = 0,
        condition_2_ratio = 0
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = FastTensorDataLoader(
            X_cat_train,
            X_cont_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        ) # Fast tensor batching, the dataloader is wrapped by cycle to make an infinite iterator
        train_iter = cycle(train_loader)

        set_seeds(seed, cuda_deterministic=True) # Fix seeds
        self.diff_model = self.diff_model.to(self.device) # Model moved to the right device
        self.diff_model.train() # The model is set to training mode

        self.ema_diff_model = ExponentialMovingAverage(
            self.diff_model.parameters(), decay=ema_decay # EMA wrapper is created for model parameters. 
        ) # This affects all model parameters (biases, weights etc within the MLP)

        self.optimizer = torch.optim.AdamW(
            self.diff_model.parameters(), lr=lr, weight_decay=0 
        ) # Optimizer is set up, which is the alogrithm updating the model params
        self.scheduler = LinearScheduler(
            num_steps_train,
            base_lr=lr,
            final_lr=1e-6,
            warmup_steps=num_steps_warmup,
            warmup_begin_lr=1e-6,
            anneal_lr=True,
        ) # Returns the custom learning schedule based on warmup steps, begin_lr, final_lr, anneal_lr and base_lr

        self.current_step = 0
        n_obs = sum_loss = 0
        train_start = time.time()

        with tqdm(
            initial=self.current_step,
            total=num_steps_train,
        ) as pbar: # Initializing a progress bar
            while self.current_step < num_steps_train:
                self.optimizer.zero_grad() # The pytorch gradients accumulate by default, so we delete them before backpropagation

                inputs = next(train_iter)
                x_cat, x_cont = (
                    input.to(self.device) if input is not None else None
                    for input in inputs
                ) # Get a batch and move it to the device

                # Compute the losses (CE, MSE, timewarp, weightnet)
                losses, _ = self.diff_model.loss_fn(x_cat, x_cont, None, cfg, y_condition_1, y_condition_2) 
                losses["train_loss"].backward() # Backpropagation is performed (to compute gradients)

                # update parameters, updating the model weights, as well as the ema for learned timewarp params and model weights
                self.optimizer.step()
                self.diff_model.timewarp_cdf.update_ema()
                self.ema_diff_model.update()

                # The total loss and progress is stored
                sum_loss += losses["train_loss"].detach().mean().item() * x_cat.shape[0]
                n_obs += x_cat.shape[0]
                self.current_step += 1
                pbar.update(1)

                if self.current_step % log_steps == 0: # print stats every log steps
                    pbar.set_description(
                        f"Loss (last {log_steps} steps): {(sum_loss / n_obs):.3f}"
                    )
                    n_obs = sum_loss = 0

                # Adjust learning rate based on scheduler
                if self.scheduler:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.scheduler(self.current_step)

        # Print time spent during training
        train_duration = time.time() - train_start
        print(f"Training took {(train_duration / 60):.2f} min.")

        # Load EMA weights into the model and set model to eval
        self.ema_diff_model.copy_to() # This replaces models current weights with EMA-smoothed ones 
        self.diff_model.eval()

        return self.diff_model # Return trained model

    def sample(self, num_samples, num_steps=200, batch_size=4096, seed=42, 
               cfg=False, probs_label_1=None, probs_label_2=None, cfg_scale =0.0):
        set_seeds(seed, cuda_deterministic=True)
        n_batches, remainder = divmod(num_samples, batch_size) # Number of batches are defined
        sample_sizes = (
            n_batches * [batch_size] + [remainder]
            if remainder != 0
            else n_batches * [batch_size]
        ) 

        x_cat_list = []
        x_cont_list = []

        for num_samples in tqdm(sample_sizes): # Looping through all batches
            cat_latents = torch.randn(
                (num_samples, self.num_cat_features, self.cat_emb_dim),
                device=self.device,
            ) # Sample gaussian noise for categorical features with shape (num_samples, num_cat_feat, cat_emb_dim)
            cont_latents = torch.randn(
                (num_samples, self.num_cont_features), device=self.device
            ) # Sample gaussian noise for cont features (num_samples, num_cont_feat)

            if cfg: # Here if cfg is true we create y_condition_1 and 2 vectors indicating the class we condition the sample on
                y_condition_1 = np.random.choice(
                    len(probs_label_1),
                    size=num_samples,
                    p=probs_label_1
                )
                y_condition_2 = np.random.choice(
                    len(probs_label_2),
                    size=num_samples,
                    p=probs_label_2
                )
                y_condition_1 = torch.tensor(y_condition_1, device=self.device).long() # NN.embedding expects type long() int as inputs
                y_condition_2 = torch.tensor(y_condition_2, device=self.device).long()
            else:
                y_condition_1 = None
                y_condition_2 = None
            
            x_cat_gen, x_cont_gen = self.diff_model.sampler(
                cat_latents, cont_latents, num_steps, cfg, y_condition_1, y_condition_2, cfg_scale
            ) # The learned reverse diffusion process. This model denoises the latents into clean synthetic data
            x_cat_list.append(x_cat_gen) # outputs of the batch stored
            x_cont_list.append(x_cont_gen)

        x_cat = torch.cat(x_cat_list).cpu() # Return final synthetic categorical and cont variables as numpy arrays
        x_cont = torch.cat(x_cont_list).cpu()

        return x_cat.long().numpy(), x_cont.numpy()
