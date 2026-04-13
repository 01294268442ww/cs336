import torch
import torch.distributed as dist
import torch.nn as nn

def broadcast_model_from_rank0(model : nn.Module) -> None:
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    
    for b in model.buffers():
        dist.broadcast(b.data, src=0)

class DDPIndividualParameters(nn.Module):
    def __init__(self, module : nn.Module) -> None:

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Distributed package is not available or not initialized.")
        super().__init__()

        self.rank = dist.get_rank()
        self.word_size = dist.get_world_size()
        self.module = module

        broadcast_model_from_rank0(self.module)

        self.handles : list[dist.Work] = []
        self._register_grad_hooks()
    
    def _register_grad_hooks(self):
        def _hook(param : nn.Parameter):
            if self.word_size == 1:
                return
            if param.grad is None:
                return
            work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(work)
        
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(_hook)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for work in self.handles:
            work.wait()
        
        if self.word_size > 1:
            for p in self.module.parameters():
                if p.grad is not None:
                    p.grad.div_(self.word_size)
        self.handles.clear()
        

