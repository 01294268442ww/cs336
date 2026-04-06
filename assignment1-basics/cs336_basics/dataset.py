import torch
from dataclasses import dataclass
import numpy as np


def data_loading(dataset, batch_size, context_length, device):
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    n = int(dataset.shape[0])
    dataset = torch.as_tensor(dataset)

    starts = torch.randint(0, n - context_length, (batch_size, ), dtype=torch.long)
    offsets = torch.arange(context_length, dtype=torch.long).unsqueeze(0)
    idx = starts.unsqueeze(1) + offsets

    inputs = dataset[idx]
    targets = dataset[idx + 1]

    inputs = inputs.to(device)
    targets = targets.to(device)

    return inputs, targets

@dataclass
class BatchState:
    pos = 0


def get_batch_sequential(x_t, batch_size, context_length, device, state, stride=None):

    if stride is None:
        stride = context_length

    n = x_t.numel()
    if n - context_length - 1 < 0:
        raise ValueError(f"Squence too short: n={n}, context_length={context_length}")
    
    start = state.pos + (batch_size - 1) * stride
    end = start + context_length + 1
    if end > n:
        state.pos = 0
        start = (batch_size - 1) * stride
        end = start + context_length + 1
    
    data = x_t[state.pos : end]
    inputs = data.as_strided(size=(batch_size, context_length), stride=(stride, 1))
    targets = data[1:].as_strided(size=(batch_size, context_length), stride=(stride, 1))

    state.pos += batch_size * stride

    # for GPU first transfer second change type
    if isinstance(device, torch.device) and device.type == "cuda":
        inputs = inputs.to(device, non_blocking=True).long()
        targets = targets.to(device, non_blocking=True).long()
    else:
        inputs = inputs.long().to(device)
        targets = targets.long().to(device)
    
    return inputs, targets


def data_loading_sequential(x, batch_size, context_length, device, state, stride=None):
    return get_batch_sequential(x, batch_size, context_length, device, state, stride)