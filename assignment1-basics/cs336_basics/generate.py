import torch
from cs336_basics.modules.utility import softmax


def top_k_sampling(logits, top_k, temperature):
    """
    logits (batch_size, vocab_size)
    """
    if top_k <= 0:
        prob = softmax(logits, dim=-1, temperature=temperature)
        return torch.multinomial(prob, num_samples=1)
    
    # topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
    # filled_logits = torch.full_like(logits, fill_value=float("-inf"))
    # filled_logits.scatter_(dim=-1, index=topk_indices, src=topk_logits)
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    filtered_logits = torch.where(logits < threshold, float("-inf"), logits)
    prob = softmax(filtered_logits, dim=-1, temperature=temperature)

    next_token = torch.multinomial(prob, num_samples=1)

    return next_token

def top_p_sampling(logits, top_p, temperature):
    """
    logits (batch_size, vocab_size)
    """

    assert 0.0 < top_p <= 1.0, "top_p must be a valid prob"

    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    sorted_prob = softmax(sorted_logits, dim=-1, temperature=temperature)
    accumulated_logits = torch.cumsum(sorted_prob, dim=-1)

    # remove sum less than top_p
    sorted_indices_to_remove = accumulated_logits > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

    filled_logits = logits.masked_fill(indices_to_remove, float("-inf"))
    prob = softmax(filled_logits, dim=-1, temperature=temperature)
    next_token = torch.multinomial(prob, num_samples=1)

    return next_token


@torch.inference_mode()
def generate(model, prompt, tokenizer, max_new_tokens, top_k=50, top_p=0.5, temperature=0.8, device=None):

    model.eval()
    if isinstance(prompt, str):
        input_ids = tokenizer.encode(prompt)
        # add batch dim
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    else:
        input_ids = prompt.unsqueeze(0)

    input_ids = input_ids.to(device)
    input_lens = input_ids.size(1)

    with torch.autocast("cuda", enabled=False):
        for _ in range(max_new_tokens):
            logits, _ = model(input_ids)
            next_token_logits = logits[:, -1, :].float()

            assert temperature > 0.0, "Temperature must be positive"
            # assert not (top_k > 0 and top_p > 0), "top_k and top_p cannot be used together"

            if top_k > 0:
                next_token_id = top_k_sampling(next_token_logits, top_k, temperature)
            elif top_k > 0.0:
                next_token_id = top_p_sampling(next_token_logits, top_p, temperature)
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if next_token_id.squeeze().item() == tokenizer.EOS_TOKEN:
                break

            input_ids = torch.concat([input_ids, next_token_id], dim=-1)
    
    input_ids = input_ids.squeeze(0)
    all_text = tokenizer.decode(input_ids.tolist())
    generate_ids = input_ids[input_lens:]
    generate_text = tokenizer.decode(generate_ids.tolist())

    model.train()

    return {
        "all_text" : all_text,
        "generated_text" : generate_text,
        "generated_ids" : generate_ids
    }