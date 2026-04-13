from __future__ import annotations

from typing import Any, Dict, Type, Iterable

import torch
import torch.distributed as dist
from torch.optim import Optimizer

class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any) -> None:

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(kwargs)

        self.all_params : list[torch.nn.Parameter] = []
        self.param_to_global_idx : dict[int, int] = {}

        self.local_optimizer : Optimizer | None = None

        super().__init__(params, defaults=self.optimizer_kwargs)
    
    def _owner_rank(self, param : torch.nn.Parameter) -> int:
        idx = self.param_to_global_idx[id(param)]
        return idx % self.world_size
    
    def _assign_and_track_params(self, group_params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
        local : list[torch.nn.Parameter] = []
        for p in group_params:
            pid = id(p)
            if pid not in self.param_to_global_idx:
                gid = len(self.all_params)
                self.all_params.append(p)
                self.param_to_global_idx[pid] = gid
            
            if self.world_size == 1 or self._owner_rank(p) == self.rank:
                local.append(p)
        
        return local
        
    def add_param_group(self, param_group: dict[str, Any]):
        if "params" not in param_group:
            raise ValueError("param_group must have 'params' key")

        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            raise TypeError("params must be Iterable of Parameter")
        
        group_params = list(params)

        local_params = self._assign_and_track_params(group_params)

        
        local_group = {k:v for k, v in param_group.items() if k != "params"}
        local_group["params"] = local_params
        
        if len(local_params) == 0:
            return
        
        super().add_param_group(param_group)

        if self.local_optimizer is None:
            self.local_optimizer = self.optimizer_cls(self.param_groups, **self.optimizer_kwargs)
        else:
            self.local_optimizer.add_param_group(local_group)

    
    @torch.no_grad()
    def step(self, closure=None, **kwargs: Any):
        loss = None

        if self.local_optimizer is not None:
            loss = self.local_optimizer.step(closure=closure, **kwargs)

        if self.world_size > 1:
            for p in self.all_params:
                dist.broadcast(p.data, src=self._owner_rank(p))
        
        return loss