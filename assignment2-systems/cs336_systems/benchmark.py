import timeit
import os
import torch
import argparse
import logging
import pandas as pd
from statistics import mean, stdev
from contextlib import nullcontext
from cs336_basics.cs336_basics.model import BasicsTransformerLM

parser = argparse.ArgumentParser(description="pass a basics Transformer model")
parser.add_argument("--d_model", type=int, default=768, help="pass dimension of model")
parser.add_argument("--d_ff", type=int, default=3072, help="pass dimension of feed-forward network")
parser.add_argument("--num_layers", type=int, default=12, help="pass layers of model")
parser.add_argument("--num_heads", type=int, default=12, help="pass head of multi-attn")
parser.add_argument("--context_length", type=int, default=256, help="pass context_length")
parser.add_argument("--rope_theta", type=float, default=10000, help="pass RoPE's theat")
parser.add_argument("--num_warmups", type=int, default=10, help="pass num_warmups")
parser.add_argument("--num_trials", type=int, default=10, help="pass num_trials")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def profile(model, inputs, targets, vocab_size, num_warmups = 10, num_trials = 10):

    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    def forward_only():
        out = model(inputs)
        return out

    def backward_only(out):
        loss = criterion(out.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

    for _ in range(num_warmups):
        out = forward_only()
        backward_only(out)

    if torch.cuda.is_available():
        torch.cuda.synchronize() # wait for CUDA threads to finish

    for trial in range(num_trials):
        """forward"""
        torch.cuda.synchronize()
        torch.cuda.memory._record_memory_history(max_entries=1000000, context="all", stacks="all")
        torch.cuda.reset_peak_memory_stats()
        out = model(inputs)
        torch.cuda.synchronize()
        fwd_alloc = torch.cuda.max_memory_allocated()
        fwd_res = torch.cuda.max_memory_reserved()
        torch.cuda.memory._dump_snapshot(f"memory_forward{"_" + str(trial)}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

        """backward"""
        loss = criterion(out.view(-1, vocab_size), targets.view(-1))
        torch.cuda.memory._record_memory_history(max_entries=1000000, context="all", stacks="all")
        torch.cuda.reset_peak_memory_stats()
        loss.backward()
        torch.cuda.synchronize()
        bwd_alloc = torch.cuda.max_memory_allocated()
        bwd_res = torch.cuda.max_memory_reserved()
        torch.cuda.memory._dump_snapshot(f"memory_backward{"_" + str(trial)}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        del loss, out
        torch.cuda.synchronize()

        """optimizer"""
        torch.cuda.synchronize()
        torch.cuda.memory._record_memory_history(max_entries=1000000, context="all", stacks="all")
        torch.cuda.reset_peak_memory_stats()
        optimizer.step()
        torch.cuda.synchronize()
        opt_alloc = torch.cuda.max_memory_allocated()
        opt_res = torch.cuda.max_memory_reserved()
        torch.cuda.memory._dump_snapshot(f"memory_optim{"_" + str(trial)}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

        print(
            f"[PEAK] forward  alloc={fwd_alloc/1e9:.3f} GB, reserved={fwd_res/1e9:.3f} GB\n"
            f"[PEAK] backward alloc={bwd_alloc/1e9:.3f} GB, reserved={bwd_res/1e9:.3f} GB\n"
            f"[PEAK] optim    alloc={opt_alloc/1e9:.3f} GB, reserved={opt_res/1e9:.3f} GB"
        )
    


def benchmark(model, inputs, targets, vocab_size, num_warmups = 10, num_trials = 10):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.

    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    def forward_only():
        out = model(inputs)
        return out

    def backward_only(out):
        loss = criterion(out.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

    for _ in range(num_warmups):
        out = forward_only()
        backward_only(out)

    if torch.cuda.is_available():
        torch.cuda.synchronize() # wait for CUDA threads to finish
    
    forward_times : list[float] = []
    backward_times : list[float] = []
    for _ in range(num_trials):
        start_time = timeit.default_timer()

        out = forward_only()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = timeit.default_timer()

        forward_times.append(end_time - start_time)

        start_time = timeit.default_timer()

        backward_only(out)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = timeit.default_timer()

        backward_times.append(end_time - start_time)

    mean_forward = mean(forward_times)
    mean_backward = mean(backward_times)

    std_forward = stdev(forward_times) if len(forward_times) > 1 else 0
    std_backward = stdev(backward_times) if len(backward_times) > 1 else 0

    return mean_forward, std_forward, mean_backward, std_backward


def benchmarking():

    args = parser.parse_args()

    logger.info(
        f"Initialize a model hyperparameters : \
        d_model: {args.d_model} d_ff: {args.d_ff} num_layers: {args.num_layers} num_heads: {args.num_heads} \
        num_warmups: {args.num_warmups} num_trials: {args.num_trials}"
    )

    vocab_size = 10000
    batch_size = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")


    model = BasicsTransformerLM(
        vocab_size=vocab_size, 
        context_length=args.context_length, 
        d_model=args.d_model, 
        d_ff=args.d_ff, 
        num_heads=args.num_heads, 
        num_layers=args.num_layers,
        rope_theta=args.rope_theta
    ).to(device)

    
    logger.info("Create a random batch of data")
    x = torch.randint(0, vocab_size, (batch_size, args.context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, args.context_length), device=device)

    # logger.info("Testing forward and backward")

    # mean_forward, std_forward, mean_backward, std_backward = benchmark(model, x, y, vocab_size, args.num_warmups, args.num_trials)

    # logger.info(f"mean_forward: {mean_forward}, std_forward: {std_forward}, mean_backward: {mean_backward}, std_backward: {std_backward}")

    profile(model, x, y, vocab_size, args.num_warmups, args.num_trials)


def main():
    benchmarking()

if __name__ == "__main__":
    main()