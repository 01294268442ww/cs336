import torch

def save_checkpoint(model, optimizer, iteration, out):
    """
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    """
    state = {
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "iteration":iteration
    }

    torch.save(state, out)

def load_checkpoint(src, model, optimizer):
    """
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    """
    state = torch.load(src)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])

    return state["iteration"]