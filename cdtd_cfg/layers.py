import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

def normalize_emb(emb, dim):
    return F.normalize(emb, dim=dim, eps=1e-20)


class FourierFeatures(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        assert (emb_dim % 2) == 0
        self.half_dim = emb_dim // 2 
        self.register_buffer("weights", torch.randn(1, self.half_dim)) 
        
    def forward(self, x):
        freqs = x.unsqueeze(1) * self.weights * 2 * np.pi 
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fouriered


class WeightNetwork(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.fourier = FourierFeatures(emb_dim)
        self.fc = nn.Linear(emb_dim, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, u):
        x = self.fourier(u)
        return self.fc(x).squeeze()

    def loss_fn(self, preds, avg_loss):
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
        max_period: int = 10000,
        n_layers: int = 2, 
        fourier: bool = False, 
        scale=16, 
    ):
        super().__init__() 
        self.dim = dim
        self.max_period = max_period
        self.n_layers = n_layers
        self.fourier = fourier

        if dim % 2 != 0: 
            raise ValueError(f"embedding dim must be even, got {dim}")

        if fourier: 
            self.register_buffer("freqs", torch.randn(dim // 2) * scale)

        layers = []
        for i in range(n_layers - 1): 
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.SiLU())
        self.fc = nn.Sequential(*layers, nn.Linear(dim, dim))

    def forward(self, timesteps):
        if not self.fourier:
            d, T = self.dim, self.max_period
            mid = d // 2
            
            fs = torch.exp(-math.log(T) / mid * torch.arange(mid, dtype=torch.float32)) 
            fs = fs.to(timesteps.device)

            args = timesteps[:, None].float() * fs[None]

            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        else:
            x = timesteps.ger((2 * torch.pi * self.freqs).to(timesteps.dtype))
            emb = torch.cat([x.cos(), x.sin()], dim=1)

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
        super().__init__()
        self.num_cont_features = num_cont_features
        self.num_cat_features = len(categories) 
        dim_out = sum(categories) + self.num_cont_features 
        self.linear = nn.Linear(dim_in, dim_out) 
        nn.init.zeros_(self.linear.weight) 
        if bias_init is None: 
            nn.init.zeros_(self.linear.bias) 
        else: 
            self.linear.bias = nn.Parameter(bias_init) 
        self.split_chunks = [self.num_cont_features, *categories] 
        self.cat_idx = 0
        if self.num_cont_features > 0:
            self.cat_idx = 1

    def forward(self, x):
        x = self.linear(x)
        out = torch.split(x, self.split_chunks, dim=-1)

        if self.num_cont_features > 0:
            cont_logits = out[0]
        else:
            cont_logits = None
        if self.num_cat_features > 0:
            cat_logits = out[self.cat_idx :]
        else:
            cat_logits = None
        
        return cat_logits, cont_logits


class PositionalEmbedder(nn.Module):
    """
    Positional embedding layer for encoding continuous features.
    Adapted from https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py#L61
    """

    def __init__(self, dim, num_features, trainable=False, freq_init_scale=0.01):
        super().__init__()
        assert (dim % 2) == 0 
        self.half_dim = dim // 2
        self.weights = nn.Parameter(
            torch.randn(1, num_features, self.half_dim), requires_grad=trainable 
        )
        self.sigma = freq_init_scale 
        bound = self.sigma * 3 
        nn.init.trunc_normal_(self.weights, 0.0, self.sigma, a=-bound, b=bound)

    def forward(self, x):
        # 
        x = rearrange(x, "b f -> b f 1") 
        freqs = x * self.weights * 2 * torch.pi 
        fourier = torch.cat((freqs.sin(), freqs.cos()), dim=-1) 
        return fourier


class CatEmbedding(nn.Module):
    """
    Feature-specific embedding layer for categorical features.
    bias = True adds a learnable bias term to each feature, which is is same across categories.
    """

    def __init__(self, dim, categories, cat_emb_init_sigma=0.001, bias=False):
        super().__init__()

        self.categories = torch.tensor(categories)  
        categories_offset = self.categories.cumsum(dim=-1)[:-1] 
        categories_offset = torch.cat(
            (torch.zeros((1,), dtype=torch.long), categories_offset) 
        )
        self.register_buffer("categories_offset", categories_offset) 
        self.dim = torch.tensor(dim)

        self.cat_emb = nn.Embedding(sum(categories), dim) 
        nn.init.normal_(self.cat_emb.weight, std=cat_emb_init_sigma) 

        self.bias = bias
        if self.bias: 
            self.cat_bias = nn.Parameter(torch.zeros(len(categories), dim))

    def forward(self, x): 
        x = self.cat_emb(x + self.categories_offset)
        if self.bias: 
            x += self.cat_bias
        x = normalize_emb(x, dim=2) * self.dim.sqrt()
        return x

    def get_all_feat_emb(self, feat_idx):
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
    def __init__(
        self,
        num_cont_features,
        cat_emb_dim,
        categories,
        proportions,
        emb_dim,
        n_layers, 
        n_units,
        act="relu",
        num_classes_1 = None, # Number of categories within both features are passed
        num_classes_2 = None
    ):
        super().__init__()

        num_cat_features = len(categories)

        self.time_emb = TimeStepEmbedding(emb_dim, fourier=False) 
        # For each label containing classes an embedding is initialized with output emb_dim and num_class inputs
        if num_classes_1 is not None: 
            self.cond_embed_1 = Label_Embedding(num_classes_1, emb_dim)
        else:
            self.cond_embed_1 = None

        if num_classes_2 is not None:
            self.cond_embed_2 = Label_Embedding(num_classes_2, emb_dim)
        else:
            self.cond_embed_2 = None

        in_dims = [emb_dim] + (n_layers - 1) * [n_units]
        out_dims = n_layers * [n_units]
        layers = nn.ModuleList()
        for i in range(len(in_dims)):
            layers.append(nn.Linear(in_dims[i], out_dims[i]))
            layers.append(nn.ReLU() if act == "relu" else nn.SiLU())
        self.fc = nn.Sequential(*layers) 
        
        dim_in = num_cont_features + num_cat_features * cat_emb_dim
        self.proj = nn.Linear(dim_in, emb_dim)

        cont_bias_init = torch.zeros((num_cont_features,))
        cat_bias_init = torch.cat(proportions).log()
        bias_init = torch.cat((cont_bias_init, cat_bias_init))

        self.final_layer = FinalLayer(
            out_dims[-1], categories, num_cont_features, bias_init=bias_init
        )

    # The dropout_ratio, the labels, and a boolean indicating whether we are fitting or sampling are passed to the forward of the mlp
    def forward(self, x_cat_emb_t, x_cont_t, time, cfg = False, y_condition_1=None, y_condition_2=None, dropout_ratio = 1.0, sample=False):
        cond_emb = self.time_emb(time)
        x = torch.concat((rearrange(x_cat_emb_t, "B F D -> B (F D)"), x_cont_t), dim=-1)   
        x = self.proj(x)
        
        if cfg:
            batch_size = x_cont_t.shape[0]

            if sample:
                if y_condition_1 is not None:
                    label_1_emb = self.cond_embed_1(y_condition_1)
                    cond_emb = cond_emb + label_1_emb
    
                if y_condition_2 is not None:
                    label_2_emb = self.cond_embed_2(y_condition_2)
                    cond_emb = cond_emb + label_2_emb
                     
            else:
                condition_1_ratio = (1 - dropout_ratio)/2
                condition_2_ratio = (1 - dropout_ratio)/2
    
                # We initialize a group mask to indicate what group the obsevation the observation belongs to
                mask = torch.rand(batch_size, device=x_cont_t.device) # Draws a random uniform value for each observation in the batch
    
                # Compute thresholds
                cutoff1 = dropout_ratio
                cutoff2 = dropout_ratio + condition_1_ratio
    
                # Apply rules
                use_label_1 = (mask >= cutoff1) & (mask < cutoff2)
                use_label_2 = (mask >= cutoff2)
    
                if y_condition_1 is not None:
                    mask_1 = y_condition_1 == -1  # True where value is -1
                    y_condition_1_fixed = torch.abs(y_condition_1)  # converts -1 to 1, leaves rest unchanged
                    label_1_emb = self.cond_embed_1(y_condition_1_fixed)
                    label_1_emb[mask_1] = 0.0  # zero out embeddings where original was -1
                    cond_emb = cond_emb + label_1_emb
    
                if y_condition_2 is not None:
                    mask_2 = y_condition_2 == -1
                    y_condition_2_fixed = torch.abs(y_condition_2)
                    label_2_emb = self.cond_embed_2(y_condition_2_fixed)
                    label_2_emb[mask_2] = 0.0
                    cond_emb = cond_emb + label_2_emb
    
            x = x + cond_emb 
            
        x = self.fc(x)
        return self.final_layer(x)

class Timewarp_Logistic(nn.Module):
    """
    Our version of timewarping with exact cdfs instead of p.w.l. functions.
    We use a domain-adapted cdf of the logistic distribution.

    timewarp_type selects the type of timewarping:
        - single (single noise schedule, like CDCD)
        - bytype (per type noise schedule)
        - all (per feature noise schedule)
    """
    

    def __init__(
        self,
        timewarp_type,
        num_cat_features, 
        num_cont_features,
        sigma_min,
        sigma_max,
        weight_low_noise=1.0, 
        decay=0.0, 
    ):
        super(Timewarp_Logistic, self).__init__()

        self.timewarp_type = timewarp_type
        self.num_cat_features = num_cat_features
        self.num_cont_features = num_cont_features
        self.num_features = num_cat_features + num_cont_features

        self.register_buffer("sigma_min", sigma_min)
        self.register_buffer("sigma_max", sigma_max)

        if timewarp_type == "single":
            self.num_funcs = 1
        elif timewarp_type == "bytype":
            self.num_funcs = 2
        elif timewarp_type == "all":
            self.num_funcs = self.num_cat_features + self.num_cont_features


        v = torch.tensor(1.01)
        logit_v = torch.log(torch.exp(v - 1) - 1) 
        self.logits_v = nn.Parameter(torch.full((self.num_funcs,), fill_value=logit_v)) 
        self.register_buffer("init_v", self.logits_v.clone()) 

        p_large_noise = torch.tensor(1 / (weight_low_noise + 1))
        
        logit_mu = torch.log(((1 / (1 - p_large_noise)) - 1)) / v
        
        self.logits_mu = nn.Parameter(
            torch.full((self.num_funcs,), fill_value=logit_mu)
        )
        self.register_buffer("init_mu", self.logits_mu.clone())

        self.logits_gamma = nn.Parameter(
            (torch.ones((self.num_funcs, 1)).exp() - 1).log()
        )

        self.decay = decay 
        logits_v_shadow = torch.clone(self.logits_v).detach()
        logits_mu_shadow = torch.clone(self.logits_mu).detach()
        logits_gamma_shadow = torch.clone(self.logits_gamma).detach()
        self.register_buffer("logits_v_shadow", logits_v_shadow)
        self.register_buffer("logits_mu_shadow", logits_mu_shadow)
        self.register_buffer("logits_gamma_shadow", logits_gamma_shadow)

    def update_ema(self):
        with torch.no_grad():  
            self.logits_v.copy_(
                self.decay * self.logits_v_shadow
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

            self.logits_v_shadow.copy_(self.logits_v)
            self.logits_mu_shadow.copy_(self.logits_mu)
            self.logits_gamma_shadow.copy_(self.logits_gamma)

    def get_params(self):
        logit_mu = self.logits_mu  
        v = 1 + F.softplus(self.logits_v) 
        scale = F.softplus(self.logits_gamma)
        return logit_mu, v, scale 

    def cdf_fn(self, x, logit_mu, v):
        "mu in (0,1), v >= 1"
        Z = ((x / (1 - x)) / logit_mu.exp()) ** (-v) 
        return 1 / (1 + Z) 

    def pdf_fn(self, x, logit_mu, v):
        Z = ((x / (1 - x)) / logit_mu.exp()) ** (-v)
        return (v / (x * (1 - x))) * (Z / ((1 + Z) ** 2))

    def quantile_fn(self, u, logit_mu, v):
        c = logit_mu + 1 / v * torch.special.logit(u, eps=1e-7)
        return F.sigmoid(c)

    def forward(self, x, invert=False, normalize=False, return_pdf=False):        
        logit_mu, v, scale = self.get_params()

        if not invert: 
            if normalize:
                scale = 1.0

            x = (x - self.sigma_min) / (self.sigma_max - self.sigma_min)

            x = torch.clamp(x, 1e-7, 1 - 1e-7)

            if self.timewarp_type == "single":
                input = x[:, 0].unsqueeze(0) 

            elif self.timewarp_type == "bytype":
                input = torch.stack((x[:, 0], x[:, -1]), dim=0) 

            elif self.timewarp_type == "all":
                input = x.T  

            if return_pdf: 
                output = (torch.vmap(self.pdf_fn, in_dims=0)(input, logit_mu, v)).T
            else:
                output = (
                    torch.vmap(self.cdf_fn, in_dims=0)(input, logit_mu, v) * scale 
                ).T

        else: 
            input = repeat(x, "b -> f b", f=self.num_funcs) 
            output = (torch.vmap(self.quantile_fn, in_dims=0)(input, logit_mu, v)).T

            if self.timewarp_type == "single":
                output = repeat(output, "b 1 -> b f", f=self.num_features)
            elif self.timewarp_type == "bytype":
                output = torch.column_stack(
                    (
                        repeat(output[:, 0], "b -> b f", f=self.num_cat_features), 
                        repeat(output[:, 1], "b -> b f", f=self.num_cont_features), 
                    )
                ) 
               
            zero_mask = x == 0.0 
            one_mask = x == 1.0
            
            output = output.masked_fill(zero_mask.unsqueeze(-1), 0.0) 
            output = output.masked_fill(one_mask.unsqueeze(-1), 1.0)

            output = output * (self.sigma_max - self.sigma_min) + self.sigma_min 
            
        return output 
    
    def loss_fn(self, sigmas, losses):

        if self.timewarp_type == "single":
            losses = losses.mean(1, keepdim=True)
        elif self.timewarp_type == "bytype":
            losses_cat = losses[:, : self.num_cat_features].mean(
                1, keepdim=True
            )
            losses_cont = losses[:, self.num_cat_features :].mean(
                1, keepdim=True
            )
            losses = torch.cat((losses_cat, losses_cont), dim=1)
      
        losses_estimated = self.forward(sigmas)

        with torch.no_grad():
            pdf = self.forward(sigmas, return_pdf=True).detach() 
            
        return ((losses_estimated - losses) ** 2) / (pdf + 1e-7)
