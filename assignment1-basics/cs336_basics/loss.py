import torch

def corss_entropy(logits, targets):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """

    logits = logits - torch.max(logits, dim=1, keepdim=True).values
    logits = logits - torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))
    loss = logits.gather(1, targets.unsqueeze(1))

    loss = -loss.mean()

    return loss

   
def perplexity(loss):
    return torch.exp(loss)