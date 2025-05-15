import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seeds(seed, cuda_deterministic=False):
    # Python, NumPy and PyTorch seeds are set
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

def cycle(dl):
    # Infinite loop in dataloader for training loop
    while True:
        for data in dl:
            yield data
            

def low_discrepancy_sampler(num_samples, device):
    """
    Inspired from the Variational Diffusion Paper (Kingma et al., 2022)
    """
    # Creates a tensor of num_samples values. They are drawn from (0,1) in a uniform way that however is non repetitive
    # This can be helpful for drawing timesteps or noise levels more evenly across the interval than with pure random draws
    single_u = torch.rand((1,), device=device, requires_grad=False, dtype=torch.float64) # random float
    return (
        single_u
        + torch.arange(
            0.0, 1.0, step=1.0 / num_samples, device=device, requires_grad=False #  creates a tensor: [0.0, 1/N, ..., (N-1)/N]
        )
    ) % 1 # Creates evenly spaced increments and wraps via mod 1


class LinearScheduler:
    # Handles learning rate schedules 
    def __init__(
        self,
        max_update, # number of training steps
        base_lr=0.1, # Target learning rate after warm up
        final_lr=0.0, # final learning rate after decay (if anneal_lr = True)
        warmup_steps=0, # how many steps to warm up
        warmup_begin_lr=0, # initial lr after the first step (during wm)
        anneal_lr=False, # Whether to linearly decay after warmup to final_lr
    ):
        self.anneal_lr = anneal_lr
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps # Not actually used anymore (but this same equation found in decrease)

    def get_warmup_lr(self, step): # This method gradually increases the lr during the warmup steps (adding an increase each step)
        increase = (
            (self.base_lr_orig - self.warmup_begin_lr)
            * float(step)
            / float(self.warmup_steps)
        )
        return self.warmup_begin_lr + increase

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.get_warmup_lr(step) # If still in warmup, return otuput of get_warmup_lr
        if (step <= self.max_update) and self.anneal_lr: # Else, if past warmup and anneal = true, we compute the decreased version
            decrease = (
                (self.final_lr - self.base_lr_orig)
                / (self.max_update - self.warmup_steps)
                * (step - self.warmup_steps)
            )
        return decrease + self.base_lr_orig if self.anneal_lr else self.base_lr_orig # If self anneal_lr is false, it returns base_lr


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Adapted from: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    # So a custom dataloader used for tensors. Avoiding the overhead of the default pytorch Dataloader which is slow with large batches

    def __init__(self, X_cat, X_cont, batch_size=32, shuffle=False, drop_last=False, y_condition_1 = None, y_condition_2 = None,):
        self.dataset_len = X_cat.shape[0] if X_cat is not None else X_cont.shape[0] # dataset length is initalized
        assert all( # All tensors need to have the same length 
            t.shape[0] == self.dataset_len for t in (X_cat, X_cont) if t is not None
        )
        self.X_cat = X_cat
        self.X_cont = X_cont
        
        self.y_condition_1 = y_condition_1
        self.y_condition_2 = y_condition_2

        self.batch_size = batch_size
        self.shuffle = shuffle # Whether to shuffle rows each epoch

        if drop_last: # Whether to drop the last batch if not divisible by batch size
            self.dataset_len = (self.dataset_len // self.batch_size) * self.batch_size

        # Calculates how many batches the dataset will contain
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1 # If there are leftovers we add one more
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len) # If shuffle is true, we generate a random permutation of indices
        else: # Else we just iterate in order
            self.indices = None
        self.i = 0 
        return self

    def __next__(self):
        if self.i >= self.dataset_len: # If we have gone through the whole dataset, stop the iterator
            raise StopIteration
        if self.indices is not None: # If we use shuffled dataset, we follow that order to fetch rows
            indices = self.indices[self.i : self.i + self.batch_size] 
            batch = {}
            batch["X_cat"] = (
                # X_cat is a (dataset_length, num_features) tensor.
                # 0 indicates to select across the 0th dimension (so rows)
                # indices is a 1D tensor (for instance 3,4,1,5,2)
                # So here we select and order the rows in the order of indices (so first the 3rd row, and 2nd last)           
                torch.index_select(self.X_cat, 0, indices) # So this output is also a tensor
                if self.X_cat is not None
                else None
            )
            batch["X_cont"] = (
                torch.index_select(self.X_cont, 0, indices)
                if self.X_cont is not None
                else None
            )
            batch["y_cond_1_batch"] = (
                torch.index_select(self.y_condition_1, 0, indices)
                if self.y_condition_1 is not None
                else None
            )
            batch["y_cond_2_batch"] = (
                torch.index_select(self.y_condition_2, 0, indices)
                if self.y_condition_2 is not None
                else None
            )

        else: # If non shuffled, we simply return the next batch (of size batch_size) directly
            batch = {}
            batch["X_cat"] = (
                self.X_cat[self.i : self.i + self.batch_size]
                if self.X_cat is not None
                else None
            )
            batch["X_cont"] = (
                self.X_cont[self.i : self.i + self.batch_size]
                if self.X_cont is not None
                else None
            )
            batch["y_cond_1_batch"] = (
                self.y_condition_1[self.i : self.i + self.batch_size]
                if self.y_condition_1 is not None
                else None
            )
            batch["y_cond_2_batch"] = (
                self.y_condition_2[self.i : self.i + self.batch_size]
                if self.y_condition_2 is not None
                else None
            )

        self.i += self.batch_size # We update the batch pointer to the next iteration

        batch = tuple(batch.values()) # We transform the dictionary (batch["X_cat"], batch["X_cont"]) into a single tuple
        return batch

    def __len__(self):
        return self.n_batches # Returns the number of epochs
