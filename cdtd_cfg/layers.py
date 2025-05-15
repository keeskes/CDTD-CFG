import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

# Important: In PyTorch, we build models by creating classes that inherit from nn.Module, which is the base class for all neural networks!!!
# Pytorch needs both init and forward, in the former we define the layers and parameters of the model. In the latter how the data flows through the layers
# Other methods (like loss_fn) can be added to the model for extra functionalities
# When calling the model like a function (so FourierFeatures(emb_dim), pytorch automatically calls the forward function).


def normalize_emb(emb, dim):
    # Simply normalizes the embeddings
    return F.normalize(emb, dim=dim, eps=1e-20)


class FourierFeatures(nn.Module):
    # An important feature of fourier transformations, is taking an input value (like a feature) and transforming them
    # This is done using waves, in this way, we can look at a feature from multiple viewpoints. 
    def __init__(self, emb_dim):
        super().__init__()
        # it wants to check that the output can be cleanly split into a sine and cosine half
        assert (emb_dim % 2) == 0
        self.half_dim = emb_dim // 2 
        # self.weights is a random vector of half length of emd_dim to create a pattern of random numbers.
        # so we create this vector of 512 values in this case 
        self.register_buffer("weights", torch.randn(1, self.half_dim)) #important!!! These are random weights unrelated to u

    def forward(self, x):
        # Here we transform the data to become a emb_dim length fourier vector (so self.weights * 2pi * log(u)* 0.25)
        # Step two both are put into cos and sin
        freqs = x.unsqueeze(1) * self.weights * 2 * np.pi 
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fouriered


class WeightNetwork(nn.Module):
    # If I understand it correctly, the task of the weightnetwork is to estimate how difficult it would be to estimate the value given u. 
    # This is done by transforming the fourier value into a single value (x) and comparing the predicted loss to the avg_loss we found. 
    def __init__(self, emb_dim):
        super().__init__()

        # Here it creates a fourier vector
        self.fourier = FourierFeatures(emb_dim)
        # Here a linear layer is created that is able to transform the fourier vector into a single importance score 
        self.fc = nn.Linear(emb_dim, 1)
        # It initializes the weights and biases = 0
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, u):
        x = self.fourier(u)
        # It returns the fc (fully connected?) layer. 
        return self.fc(x).squeeze()

    def loss_fn(self, preds, avg_loss):
        # learn to fit expected average loss
        # preds is the output of the forwards function
        # avg_loss is the average loss over all features at a single noise level u for a corrupted sample
        # So to summarize, we want to create a mapping from noise level u to how hard it is to learn samples at that u
        # u -> fourier embedding -> weightnetwork -> predicted loss
        # We do not want to overestimate or underestimate the difficulty level, because overestimating makes us focus too much on easy cases
        # In the other case it is the opposite. In either way misallocating training efforts.
        # So later in the model we can simply use, the weight network says it should be high here, usually it is very difficult.
        # Therefore we should downweight the loss. If it is usually very easy we should upweight it.
        # once the value below is clsoe to 0, meaning preds =avg_loss, the model has a good idea of how much noise u offers.
        # The model therefore is being trained the right amount on this noise level. 
        # So to summarize, the weightnetwork learns the difficulty of u to steer the training into the right direction
        return (preds - avg_loss) ** 2


class TimeStepEmbedding(nn.Module):
    """
    Layer that embeds diffusion timesteps.

     Args:
        - dim (int): the dimension of the output.
        - max_period (int): controls the minimum frequency of the embeddings.
        - n_layers (int): number of dense layers
        - fourer (bool): whether to use random fourier features as embeddings
    """

    def __init__(
        self,
        dim: int,
        max_period: int = 10000, # The time it takes for a complete cycle of the wave to occur
        n_layers: int = 2, # Number of fully connected layers created the embedding
        fourier: bool = False, # Whether we use fourier instead of sinusoidal encodings
        scale=16, # Frequency scaling the fourier embeddings
    ):
        super().__init__() # Here we store the parameters and initializes
        self.dim = dim
        self.max_period = max_period
        self.n_layers = n_layers
        self.fourier = fourier

        if dim % 2 != 0: # makes sure the dimensions are even (again even number of cos and sin)
            raise ValueError(f"embedding dim must be even, got {dim}")

        if fourier: # If using fourier, again we need to create dim/2 random weights, multiplied by a scale
            self.register_buffer("freqs", torch.randn(dim // 2) * scale)

        layers = []
        for i in range(n_layers - 1): #for all layers except the output it puts a linear layer + 1 SiLu activation
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.SiLU()) # SiLu often outperforms relu
        self.fc = nn.Sequential(*layers, nn.Linear(dim, dim)) # finally it combines all these layers with a final linear layer 
        # nn.sequential runs all these layers in order

    def forward(self, timesteps):
        if not self.fourier:
            d, T = self.dim, self.max_period
            mid = d // 2 # splits into d / 2 frequencies
            # These are logarithmatically spaced frequencies, from 1 to 1/ max_period (T). This is how transformers space their frequencies
            # If we were to rewrite the expression below it would look like: [1.0, T^(-1/mid), T^(-2/mid), ..., T^(-(mid-1)/mid)]
            fs = torch.exp(-math.log(T) / mid * torch.arange(mid, dtype=torch.float32)) 
            fs = fs.to(timesteps.device)
            # Here we perform a outer product to get a [B, dim] embedding (B = batch size, dim = embedding size)
            args = timesteps[:, None].float() * fs[None]
            # Now we apply the cos and sin transformation for each product. The intuition is that each frequency reacts differently to time
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        else:
            # If it is fourier, we simply use the fourier transformation as we had seen beforehand
            x = timesteps.ger((2 * torch.pi * self.freqs).to(timesteps.dtype))
            emb = torch.cat([x.cos(), x.sin()], dim=1)

        # Lastly we return the embedding through fc defined in the forward to make the embeddings more useful.  
        # This is a layer that transforms the data from size emb to size emb (so not like weightnetwork where it goes to 1)
        return self.fc(emb)

class Label_Embedding(nn.Module):
    """
    Embeds an integer class labels into a vector of size embedding_dim.
    """
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

    
class FinalLayer(nn.Module):
    """
    Final layer that predicts logits for each category for categorical features
    and scalers for continuous features.
    """

    def __init__(self, dim_in, categories, num_cont_features, bias_init=None):
        #bias_init can initialize the output layer's bias (for log-likelihood calibration)
        super().__init__()
        self.num_cont_features = num_cont_features
        self.num_cat_features = len(categories) # important, categories is a vector containing the number of categories per cat feature
        dim_out = sum(categories) + self.num_cont_features # So summing it gives for example (2,5) = 7 total categories
        self.linear = nn.Linear(dim_in, dim_out) # importantly, it creates a logit value for each category (summing to 1 per feature)
        nn.init.zeros_(self.linear.weight) # So starting out with linear layer weights = 0
        if bias_init is None: # if bias init is set to none, all biases are initialized = 0
            nn.init.zeros_(self.linear.bias) 
        else: # Else the linear.bias = bias_init 
            # by using nn.parameter it is added as a learnable parameter to the model. 
            # So the bias is overwritten by the bias_init.
            self.linear.bias = nn.Parameter(bias_init) 
        # The output feature will be split into chuncks (for each feature individually)
        # So the split chunks is the number of cont features, followed by each of the chunks
        self.split_chunks = [self.num_cont_features, *categories] 
        # The ind of categorical features in initialized at 0
        self.cat_idx = 0
        # Since all the numerical features come first, it sets the index to 1 if continuous features are present
        if self.num_cont_features > 0:
            self.cat_idx = 1

    def forward(self, x):
        # The model’s final hidden vector is fed into the linear layer to get raw output values
        x = self.linear(x)
        # The output is split into the chunk sizes (so fist part are all continuous outputs, rest is for each cat feature)
        out = torch.split(x, self.split_chunks, dim=-1)

        # If it contains cont_features, the first batch is saved as cont_output, else nothing
        if self.num_cont_features > 0:
            cont_logits = out[0]
        else:
            cont_logits = None
        # Then these are filtered out to get the cat outputs for each feature
        if self.num_cat_features > 0:
            cat_logits = out[self.cat_idx :]
        else:
            cat_logits = None
        
        # The final output consists of a logit categorical tensors, and predicted values of continuous variables
        return cat_logits, cont_logits


class PositionalEmbedder(nn.Module):
    """
    Positional embedding layer for encoding continuous features.
    Adapted from https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py#L61
    """

    def __init__(self, dim, num_features, trainable=False, freq_init_scale=0.01):
        super().__init__()
        assert (dim % 2) == 0 # again needs to be even to embed a continuous feature into sin and cosin
        self.half_dim = dim // 2
        self.weights = nn.Parameter(
            torch.randn(1, num_features, self.half_dim), requires_grad=trainable # creating random feature again with dim (1,n_con_feat,d/2)
        )
        self.sigma = freq_init_scale # A scale is set 
        bound = self.sigma * 3 # A small range is created for a truncated distribution
        # The purpose of the original random values was just to create a right shape.
        # The values are replaced by very small truncated weights close to 0
        nn.init.trunc_normal_(self.weights, 0.0, self.sigma, a=-bound, b=bound)

    def forward(self, x):
        # 
        x = rearrange(x, "b f -> b f 1") # Adding third dimension to the tensor
        freqs = x * self.weights * 2 * torch.pi # Self.weights = "1 F D/2" -> freqs = "B F D/2". 
        # So now per batch, feature, we have distinct learnable weights
        fourier = torch.cat((freqs.sin(), freqs.cos()), dim=-1) #"B F D" now by applying sin and cos and appending them
        return fourier


class CatEmbedding(nn.Module):
    """
    Feature-specific embedding layer for categorical features.
    bias = True adds a learnable bias term to each feature, which is is same across categories.
    """

    def __init__(self, dim, categories, cat_emb_init_sigma=0.001, bias=False):
        super().__init__()
        # categories again is this same list structured like [2,5,3]
        # dim is the output dimensions
        # cat_emd_init_sigma = stddev for initialization of embeddings
        # If bias = true some learnable bias per feature is added (same for all categories?)

        self.categories = torch.tensor(categories)   # We transform the categories into a tensor
        categories_offset = self.categories.cumsum(dim=-1)[:-1] 
        categories_offset = torch.cat(
            (torch.zeros((1,), dtype=torch.long), categories_offset) # This is used to then store where each feature begins
        )
        self.register_buffer("categories_offset", categories_offset) # So feature 3 starts at index 9
        self.dim = torch.tensor(dim)

        self.cat_emb = nn.Embedding(sum(categories), dim)  # A singe matrix is created to save all embeddings across all feat
        nn.init.normal_(self.cat_emb.weight, std=cat_emb_init_sigma) # This is initialized with a very small norm distribution

        self.bias = bias
        if self.bias: # If bias = true, we create a bias vector for each feature
            self.cat_bias = nn.Parameter(torch.zeros(len(categories), dim))

    def forward(self, x): # x has shape Batch x features. Each entry is an integer representing the category ID.
        # Since both categories contain overlapping features indices (0 can be female and white) we need to transform them
        # So the offset is computed, which equal to the cum of all cat in previous features to transform this.
        # Here the x transformed with the offset is fed into the embedding, leading to B F D output
        x = self.cat_emb(x + self.categories_offset)
        if self.bias: # Feature specific bias added if applicable
            x += self.cat_bias
        # l2 normalize embedding
        # So the embeddings are normalized across the last dim, and then multiplied by the sqrt of the dimensions
        # So a specific feature embedding is normalized over sqrt(SSR of all embedding values). 
        # Example : original:  [0.2, 0.5, -0.1] -> L2 norm = sqrt(0.04 + 0.25 + 0.01) = sqrt(0.3) ≈ 0.5477
        # -> normalized = [0.2 / 0.5477, 0.5 / 0.5477, -0.1 / 0.5477] ≈ [0.365, 0.913, -0.183] -> * √3 ≈ 1.732
        x = normalize_emb(x, dim=2) * self.dim.sqrt()
        return x

    def get_all_feat_emb(self, feat_idx):
        # Help method that can be used to retrieve all embeddings for a specific feature. Exactly the same as the forward function
        emb_idx = (
            torch.arange(self.categories[feat_idx], device=self.cat_emb.weight.device)
            + self.categories_offset[feat_idx]
        )
        x = self.cat_emb(emb_idx)
        if self.bias:
            x += self.cat_bias[feat_idx]
        x = normalize_emb(x, dim=1) * self.dim.sqrt()
        return x


class MLP(nn.Module):
    """
    TabDDPM-like architecture for both continuous and categorical features.
    Used for TabDDPM and CDTD.
    """
    # The MLP is used in the denoising step, taking in noised cat and cont features, and uses timestep embedding to condition on the step
    def __init__(
        self,
        num_cont_features, # num cont features
        cat_emb_dim, # dimension of cat feature embeddings
        categories, # list of category count per feature
        proportions, # list of value frequencies for log-bias initialization
        emb_dim, # embedding dimension for MLP (after projection)
        n_layers, # number of layers
        n_units, # size of layers
        act="relu", #activation function
        num_classes_1 = None,
        num_classes_2 = None
    ):
        super().__init__()

        num_cat_features = len(categories)
        # embedding created for timestep t, used to condition network each step on time
        self.time_emb = TimeStepEmbedding(emb_dim, fourier=False) 
        if num_classes_1 is not None:
            self.cond_embed_1 = Label_Embedding(num_classes_1, emb_dim)
        else:
            self.cond_embed_1 = None

        if num_classes_2 is not None:
            self.cond_embed_2 = Label_Embedding(num_classes_2, emb_dim)
        else:
            self.cond_embed_2 = None

        in_dims = [emb_dim] + (n_layers - 1) * [n_units] # Input dimension defined for first layer
        out_dims = n_layers * [n_units] # Then afterwards we use n layers with n units each
        layers = nn.ModuleList()
        for i in range(len(in_dims)): # Sequence of linear and relu layers with respective in and out dimensions defined
            layers.append(nn.Linear(in_dims[i], out_dims[i]))
            layers.append(nn.ReLU() if act == "relu" else nn.SiLU())
        self.fc = nn.Sequential(*layers) # self.fc is now defined as this sequence of layers, meaning we can now call self.fc(x)
        
        # Projection layer is created, using input of the transformed features, and with emb.dim output. 
        # It is used to transform the data representation we have at step t into a good representation of emb_dim that can enter the MLP
        dim_in = num_cont_features + num_cat_features * cat_emb_dim
        self.proj = nn.Linear(dim_in, emb_dim)

        # The final layer is initialized and called to output logits for categories and values for cont. 
        cont_bias_init = torch.zeros((num_cont_features,))
        cat_bias_init = torch.cat(proportions).log()
        bias_init = torch.cat((cont_bias_init, cat_bias_init))

        self.final_layer = FinalLayer(
            out_dims[-1], categories, num_cont_features, bias_init=bias_init
        )

    def forward(self, x_cat_emb_t, x_cont_t, time, cfg = False, y_condition_1=None, y_condition_2=None,
               dropout_ratio = 1.0, condition_1_ratio = 0.0, condition_2_ratio = 0.0):
        
        cond_emb = self.time_emb(time) # Time is input to create conditional embedding
        if cfg:
            batch_size = x_cont_t.shape[0]
            
            # Rescaling the ratios so they sum up to 1
            dropout_ratio = dropout_ratio / (dropout_ratio + condition_1_ratio + condition_2_ratio)
            condition_1_ratio = condition_1_ratio / (dropout_ratio + condition_1_ratio + condition_2_ratio)
            condition_2_ratio = condition_2_ratio / (dropout_ratio + condition_1_ratio + condition_2_ratio)
            
            # We initialize a group mask to indicate what group the obsevation the observation belongs to
            mask = torch.rand(batch_size, device=x_cont_t.device) # Draws a random uniform value for each observation in the batch

            # Compute thresholds
            cutoff1 = dropout_ratio
            cutoff2 = dropout_ratio + condition_1_ratio

            # Apply rules
            use_label_1 = (mask >= cutoff1) & (mask < cutoff2)
            use_label_2 = (mask >= cutoff2)

            if y_condition_1 is not None:
                if y_condition_1.shape[0] != batch_size:
                    raise ValueError(
                    f"y_condition_1 shape mismatch: expected ({batch_size},), "
                    )                    
                y_condition_1_safe = y_condition_1.clone()
                y_condition_1_safe[~use_label_1] = 0   # ✅ FIX
                label_1_emb = self.cond_embed_1(y_condition_1_safe)
                label_1_emb[~use_label_1] = 0.0
                cond_emb = cond_emb + label_1_emb

            if y_condition_2 is not None:
                if y_condition_2.shape[0] != batch_size:
                    raise ValueError(
                    f"y_condition_2 shape mismatch: expected ({batch_size},), "
                    )                     
                y_condition_2_safe = y_condition_2.clone()
                y_condition_2_safe[~use_label_2] = 0   # ✅ FIX
                label_2_emb = self.cond_embed_2(y_condition_2_safe)
                label_2_emb[~use_label_2] = 0.0
                cond_emb = cond_emb + label_2_emb

        # x_cont_t = B C , x_cat_ebt_t = B F D, so here they are combined. First by flattening cat to B, (FxD). So we get B, (FxD) + C
        x = torch.concat((rearrange(x_cat_emb_t, "B F D -> B (F D)"), x_cont_t), dim=-1) 
        x = self.proj(x) + cond_emb # x is projected into B, emb_dim. Then The cond_embedding is added, also shaped B, emb_dim.
        x = self.fc(x) # The B, emb_dim data is passed through the MLP
        
        return self.final_layer(x) # The output of the final layer is returned (cat_logits and cont_pred)

class Timewarp_Logistic(nn.Module):
    """
    Our version of timewarping with exact cdfs instead of p.w.l. functions.
    We use a domain-adapted cdf of the logistic distribution.

    timewarp_type selects the type of timewarping:
        - single (single noise schedule, like CDCD)
        - bytype (per type noise schedule)
        - all (per feature noise schedule)
    """
    
    # Defines and learns a noise schedule (a mapping from 𝑢 ∈ [0,1] to noise levels σ) using a logistic CDF shape. 
    # This process is adaptive based on loss and feature type

    def __init__(
        self,
        timewarp_type, # single, bytype, all (same noise schedule for all features, by type, or different per feature)
        num_cat_features, 
        num_cont_features,
        sigma_min,
        sigma_max,
        weight_low_noise=1.0, # hyperparameter denoting desired ratio between high and low noise samples. 
        # So if 9, than we want 9x more weight on low noise samples
        decay=0.0, # used for average smoothing of learned parameters (if > 0) to stabilize the training.
    ):
        super(Timewarp_Logistic, self).__init__()

        self.timewarp_type = timewarp_type
        self.num_cat_features = num_cat_features
        self.num_cont_features = num_cont_features
        self.num_features = num_cat_features + num_cont_features

        # bounds defined for min-max scaling of the logistic CDF output
        self.register_buffer("sigma_min", sigma_min)
        self.register_buffer("sigma_max", sigma_max)

        # Number of seperate logistic CDFs need to be learned
        if timewarp_type == "single":
            self.num_funcs = 1
        elif timewarp_type == "bytype":
            self.num_funcs = 2
        elif timewarp_type == "all":
            self.num_funcs = self.num_cat_features + self.num_cont_features

        # Learnable parameters of the logistics function are initialized
        # The transformation we perform is: sigma(u) = sigmoid(μ + (1 / v) * logit(u))
        # Here:
        # - μ (mu) controls the midpoint (where the steepest change occurs)
        # - v controls the slope of the transition 
        # (the slope is steeper with smaller 1/v, so the further v from 1, the faster the change in sigma with small change in u)
        # - logit(u) = log(u / (1 - u)) maps u from [0,1] → ℝ. So 
        # - sigmoid ensures sigma(u) ∈ (0,1)


        # So this function effectively maps a uniform distribution over u into a learnable distribution over noise levels.
        # It allows the model to learn how to allocate training effort across different noise intensities.
        v = torch.tensor(1.01) # steepness
        logit_v = torch.log(torch.exp(v - 1) - 1) # Transforming v in a way that it is always (0, 1)
        self.logits_v = nn.Parameter(torch.full((self.num_funcs,), fill_value=logit_v)) 
        self.register_buffer("init_v", self.logits_v.clone()) # We nog get v = 1 + softplus(logit_v) ≈ 1.01, which makes sure > 1
        # This initial value of 1.01 makes for a gentle initial slope, making sure the model does not overfocus on certain values early

        # Mu determines where the transition/ midpoint happens on the (0, 1) range.         
        # The goal is to decide whether the model focusses more on easy or hard examples. 
        p_large_noise = torch.tensor(1 / (weight_low_noise + 1)) # Expresses the probability for high noise 
        
        # We want to find the value of μ such that: sigmoid(μ) = 1 - p_large_noise
        # This places the sigmoid’s midpoint so that only p_large_noise proportion of u-values will map to high noise levels.
        #   Since sigma(u) = sigmoid(μ + (1/v) * logit(u)), we calculate mu using the inverse: μ = (1/v) * logit(1 - p_large_noise)
        logit_mu = torch.log(((1 / (1 - p_large_noise)) - 1)) / v
        
        # logit_mu is used to initialize mu for each timewarping function (depending on the quantity of functions we have)
        self.logits_mu = nn.Parameter(
            torch.full((self.num_funcs,), fill_value=logit_mu)
        )
        self.register_buffer("init_mu", self.logits_mu.clone())

        # init gamma, scaling parameter applied to the output of the CDF, is set to 1. Again we use a trick to ensure gamma > 0
        self.logits_gamma = nn.Parameter(
            (torch.ones((self.num_funcs, 1)).exp() - 1).log()
        )

        self.decay = decay 
        # for ema we save the initial values of v, mu and gamma. 
        # furthermore all values have shadow copier that are updated more slowly.
        logits_v_shadow = torch.clone(self.logits_v).detach()
        logits_mu_shadow = torch.clone(self.logits_mu).detach()
        logits_gamma_shadow = torch.clone(self.logits_gamma).detach()
        self.register_buffer("logits_v_shadow", logits_v_shadow)
        self.register_buffer("logits_mu_shadow", logits_mu_shadow)
        self.register_buffer("logits_gamma_shadow", logits_gamma_shadow)

    def update_ema(self):
        # Method used to update the exponential moving average of logits_v, logits_mu and logits_gamma
        # Gradient tracking is turned off since we are not training here but just doing parameter updates.
        # Tracking is usually automatically turned on for backpropagation, but since we are not learning anything through the data
        # Instead we are manually updating model parameters using the decay
        with torch.no_grad():  
            self.logits_v.copy_(
                self.decay * self.logits_v_shadow # self.decay of the old value is kept
                # and 1 - self.decay of the new value is added, reducing the impact of a very noisy gradient
                # This smooths the evolution of the noise schedule
                + (1 - self.decay) * self.logits_v.detach() 
            )
            self.logits_mu.copy_(
                self.decay * self.logits_mu_shadow
                + (1 - self.decay) * self.logits_mu.detach()
            )
            self.logits_gamma.copy_(
                self.decay * self.logits_gamma_shadow
                + (1 - self.decay) * self.logits_gamma.detach()
            )
            # After updating the EMA value, we also update the shadow buffer to match it
            self.logits_v_shadow.copy_(self.logits_v)
            self.logits_mu_shadow.copy_(self.logits_mu)
            self.logits_gamma_shadow.copy_(self.logits_gamma)

    def get_params(self):
        logit_mu = self.logits_mu  # So far we computed the log of mu, so we need to transform it by ln(mu / (1-mu))
        v = 1 + F.softplus(self.logits_v)  # v > 1
        scale = F.softplus(self.logits_gamma) # gamma > 0
        return logit_mu, v, scale # they are now returned as usable parameters for the CDF

    def cdf_fn(self, x, logit_mu, v):
        "mu in (0,1), v >= 1"
        Z = ((x / (1 - x)) / logit_mu.exp()) ** (-v) 
        return 1 / (1 + Z) # Formula to compute and return the CDF, transforming x into (0, 1).
    # So output is So close to 0 with low x, close to 1 high x, and steeper depending on v.

    def pdf_fn(self, x, logit_mu, v):
        # pdf is computed (which is needed as the slope of the cdf function later on)
        Z = ((x / (1 - x)) / logit_mu.exp()) ** (-v)
        return (v / (x * (1 - x))) * (Z / ((1 + Z) ** 2))

    def quantile_fn(self, u, logit_mu, v):
        # Inverse of the cdf, taking in uniform u, and outputting logistic shaped value c
        c = logit_mu + 1 / v * torch.special.logit(u, eps=1e-7)
        #  Transformed into (0,1) range by using sigmoid
        return F.sigmoid(c)

    def forward(self, x, invert=False, normalize=False, return_pdf=False):
        # If invert = false applies time warping cdf, if true it reverts it. 
        # If return_pdf = True it returns pdf instead of cdf. 
        
        logit_mu, v, scale = self.get_params() # Retrieves learned parameters

        if not invert: # This block only applies if returning the CDF
            if normalize:
                scale = 1.0 # If normalize is true, we do not scale the output (we override the scale we retrieved from get_params)

            # can have more sigmas than cdfs
            x = (x - self.sigma_min) / (self.sigma_max - self.sigma_min) # We normalize the noise levels to [0, 1]

            # ensure x is never 0 or 1 to ensure robustness (making sure 1/x and log(x) is possible)
            x = torch.clamp(x, 1e-7, 1 - 1e-7)

            if self.timewarp_type == "single":
                # all sigmas are the same so just take first one
                input = x[:, 0].unsqueeze(0) # So if single we use one curve for everything, so we pick the first

            elif self.timewarp_type == "bytype":
                # first sigma belongs to categorical feature, last to continuous feature
                input = torch.stack((x[:, 0], x[:, -1]), dim=0) # Here input becomes 2,B (first and last feature)

            elif self.timewarp_type == "all":
                input = x.T  # (num_features, batch), so each feature has its own timewarp curve

            # Depending on the request, the pdf or cdf is teruned. 
            # These function is run separately for each log param set
            if return_pdf: 
                output = (torch.vmap(self.pdf_fn, in_dims=0)(input, logit_mu, v)).T
            else:
                output = (
                    torch.vmap(self.cdf_fn, in_dims=0)(input, logit_mu, v) * scale 
                ).T

        else: # In the case we call the inverse we return the noise level sigma from a uniform value
            # u is extended so every feature/ feature type or all data (depending on num_funcs) can have its own version of the input
            input = repeat(x, "b -> f b", f=self.num_funcs) 
            # Each noise level is mapped through its own logistic inverse, returning sigma
            output = (torch.vmap(self.quantile_fn, in_dims=0)(input, logit_mu, v)).T

            if self.timewarp_type == "single":
                output = repeat(output, "b 1 -> b f", f=self.num_features) # Single = same value for all features
            elif self.timewarp_type == "bytype":
                output = torch.column_stack(
                    (
                        repeat(output[:, 0], "b -> b f", f=self.num_cat_features), # One for cateogircal
                        repeat(output[:, 1], "b -> b f", f=self.num_cont_features), # One for numerical
                    )
                ) # Importantly, the all case already has the right shape
               
            zero_mask = x == 0.0 # Creating booleans = true if x = 0 (or 1)
            one_mask = x == 1.0
            
            # Sets all values to 0.0 and 1.0, meaning if u = 0.0 or 1.0 sigma is set to 0.0 or 1.0, avoiding unstable computation
            output = output.masked_fill(zero_mask.unsqueeze(-1), 0.0) 
            output = output.masked_fill(one_mask.unsqueeze(-1), 1.0)

            # The sigma levels are rescaled to their original scale instead of [0, 1]
            output = output * (self.sigma_max - self.sigma_min) + self.sigma_min 
            
        # This output is either the importance transformed sigma (for weighting loss), PDF (normalization), sampled sigma (from u)
        return output 
    
    def loss_fn(self, sigmas, losses):
        # losses and sigmas have shape (B, num_features), so per feature and sample. Losses are the actually observed losses (CE or MSE)

        if self.timewarp_type == "single":
            # fit average loss (over all feature) since we train a single timewarping curve
            losses = losses.mean(1, keepdim=True)  # (B,1)
        elif self.timewarp_type == "bytype":
            # fit average loss over cat and over cont features separately
            losses_cat = losses[:, : self.num_cat_features].mean(
                1, keepdim=True
            )  # (B,1)
            losses_cont = losses[:, self.num_cat_features :].mean(
                1, keepdim=True
            )  # (B,1)
            losses = torch.cat((losses_cat, losses_cont), dim=1)

        # losses now has shape, B, num_funcs 
      
        losses_estimated = self.forward(sigmas) # expected losses are computed using sigma

        with torch.no_grad():
            # pdf is retrieved for the weighting loss, giving the density of sigma.
            # This represents how much attention the model should be paying in this region
            pdf = self.forward(sigmas, return_pdf=True).detach() 
            
        # We compute a squared error loss between the predicted and observed average loss per sample
        # We divide this by the pdf so samples with low density (rare or extreme noise levels) are not not overpenalized
        return ((losses_estimated - losses) ** 2) / (pdf + 1e-7)
