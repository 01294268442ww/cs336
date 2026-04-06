import torch
import torch.nn as nn
import gc
import random
import numpy as np


class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)
    


# default temperature is 1
# if temperature > 1 distribution more smooth
# else more sharp
def softmax(logits, dim=-1, temperature=1.0):
    max_logits = torch.max(logits, dim=dim, keepdim=True).values
    exp_logits = torch.exp((logits - max_logits) / temperature)
    sum_logits = torch.sum(exp_logits, dim=dim, keepdim=True)

    return exp_logits / sum_logits


def clear_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device : {device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device: {device}")
    return device
        

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)