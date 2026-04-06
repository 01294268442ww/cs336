from collections.abc import Callable
from typing import Optional
import torch
import math

def cosine_annealing_lr(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters,
):
    # 1. warmup
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    # 2. cosine decay
    if it <= cosine_cycle_iters:
        decay_iters = max(1, cosine_cycle_iters - warmup_iters)
        progress = (it - warmup_iters) / decay_iters
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)

    # 3. after decay
    return min_learning_rate


def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
     
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm_sq += param_norm ** 2
    
    total_norm = total_norm_sq ** 0.5
    clip_coef = max_l2_norm / (total_norm + eps)

    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef)



class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure : Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise ValueError(f"betas must be a tuple of length 2 , got: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2 value: {beta2}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

            
                with torch.no_grad():
                    p -= lr * weight_decay * p

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                step_size = lr / bias_correction1

                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
            
                with torch.no_grad():
                    p -= step_size * exp_avg / denom

        return loss
    
