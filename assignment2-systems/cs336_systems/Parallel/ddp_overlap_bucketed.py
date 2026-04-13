import torch
import torch.nn as nn
import torch.distributed as dist

def broadcast_model_from_rank0(model : nn.Module) -> None:
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    
    for b in model.buffers():
        dist.broadcast(b.data, src=0)


class DDPOverlapBucketParameter(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Distributed package is not available or initialized")

        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.module = module

        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.current_size_bytes = 0

        broadcast_model_from_rank0(self.module)

        self.bucket : list[tuple[torch.Tensor, torch.Size, int]] = []
        self.handles : list[tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Size, int]], dist.Work]] = []
        
        self._register_grad_hooks()

    def _register_grad_hooks(self):
        def _hook(param: nn.Parameter):
            if self.world_size == 1:
                return
            if param.grad is None:
                return

            grad = param.grad
            grad_size_bytes = grad.numel() * grad.element_size()

            if self.current_size_bytes + grad_size_bytes > self.bucket_size_bytes:
                self._flush_bucket()
            
            self.bucket.append((grad, grad.shape, self.current_size_bytes // 4))
            self.current_size_bytes += grad_size_bytes

        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(_hook)

    def _flush_bucket(self):
        if not self.bucket:
            return

        grad_to_concat = [g.reshape(-1) for g, _, _ in self.bucket]
        bucket_tensor = torch.cat(grad_to_concat)

        work = dist.all_reduce(bucket_tensor, op=dist.ReduceOp.SUM, async_op=True)

        self.handles.append((bucket_tensor, self.bucket.copy(), work))

        self.bucket.clear()
        self.current_size_bytes = 0
        

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        self._flush_bucket()

        for merged_tensor, grad_info, work in self.handles:
            work.wait()

            for grad, gard_shape, start_pos in grad_info:
                numel = grad.numel()

                syn_grad = merged_tensor[start_pos : start_pos + numel].view(gard_shape)

                grad.copy_(syn_grad)

        if self.world_size > 1:
            for p in self.module.parameters():
                if p.grad is not None:
                    p.grad.div_(self.world_size)

        self.handles.clear()