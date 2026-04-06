import torch
import os
import numpy as np
from tqdm import trange
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.generate import generate
from cs336_basics.loss import corss_entropy, perplexity
from cs336_basics.optm.Optimizer import gradient_clipping, cosine_annealing_lr
from cs336_basics.dataset import data_loading_sequential, BatchState
from cs336_basics.modules.utility import clear_memory
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint



@ torch.no_grad()
def eval_model(model, train_config):

    model.eval()

    eval_loss = 0.0
    eval_perplexity = 0.0

    original_data = np.memmap(
        train_config.eval_data_path,
        mode="r+",
    )

    x = torch.from_numpy(original_data)

    total_tokens = len(original_data)
    num_eval_batchs = total_tokens // (train_config.batch_size * train_config.max_seq_len)

    state = BatchState()

    with torch.no_grad():
        for _ in trange(num_eval_batchs):
            inputs, targets = data_loading_sequential(
                x, 
                train_config.batch_size, 
                train_config.max_seq_len, 
                device=next(model.parameters()).device, 
                state=state
            )

            logits, aux = model(inputs)

            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = corss_entropy(logits, targets)

            eval_loss += loss.item()
            eval_perplexity += perplexity(loss).item()

    eval_loss = torch.tensor(eval_loss / num_eval_batchs)
    eval_perplexity = torch.tensor(eval_perplexity / num_eval_batchs)

    model.train()

    return eval_loss, eval_perplexity


def train(model, optimizer, train_config):
    tokenizer = Tokenizer.from_files(train_config.vocab_path, train_config.merges_path)

    original_data = np.memmap(
        train_config.train_data_path,
        mode="r+"
    )

    x = torch.from_numpy(original_data)
    best_eval_loss = float("inf")

    state = BatchState()

    for step in trange(train_config.num_steps):

        inputs, targets = data_loading_sequential(x, train_config.batch_size, train_config.max_seq_len, train_config.device, state)

        if torch.isnan(inputs).any():
            print("NaN in inputs")

        print("input range:", inputs.min().item(), inputs.max().item())

        if inputs.max() >= 10000:
            raise ValueError("Input token out of vocab range!")

        logits, aux = model(inputs)

        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        loss = corss_entropy(logits, targets)

        if train_config.use_moe:
            z_loss_scaled = aux["z_loss_scaled"]
            moe_layers = aux["moe_layers"]

            loss = loss + (z_loss_scaled / moe_layers)

            lb_loss = aux["lb_loss_scaled"]

            loss = loss + (lb_loss / moe_layers)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        gradient_clipping(model.parameters(), train_config.max_grad_norm)

        lr = cosine_annealing_lr(
            step, 
            train_config.max_lr, 
            train_config.min_lr, 
            train_config.warmup_steps, 
            train_config.num_steps - train_config.warmup_steps
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        optimizer.step()

        print(
            f"Step {step + 1} / {train_config.num_steps}, Loss: {loss.item():.4f}, LR: {lr:.6f}"
        )

        if train_config.use_moe and  train_config.log_moe_every > 0 and (step + 1) % train_config.log_moe_every == 0:
            print(f"use MoE and token_per_expert is : {aux["token_per_expert"]}")

        if train_config.eval_log_interval > 0 and (step + 1) % train_config.eval_log_interval == 0:
            del inputs, targets, logits, loss
            clear_memory()

            print("Evaluting model...")

            eval_loss, eval_perplexity = eval_model(model, train_config)
            print(
                f"Eval Loss: {eval_loss.item():.4f}, Eval Perplexity: {eval_perplexity.item():.4f}", "blue"
            )

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                print(f"New best eval loss: {best_eval_loss:.4f}")
                save_path = os.path.join(
                    train_config.save_checkpoint_dir,
                    train_config.model_name,
                    f"best_model_step_{step + 1}.pt"
                )

                save_checkpoint(model, optimizer, step, save_path)

        if train_config.sampling_log_interval > 0 and (step + 1) % train_config.sampling_log_interval == 0:
            out = generate(model, "Once upon a time", tokenizer, max_new_tokens=256, device=train_config.device)

            generate_text = out["generated_text"]
            print(f"Generate text at step {step + 1}")
            print("Once upon a time", end="")
            print(generate_text)