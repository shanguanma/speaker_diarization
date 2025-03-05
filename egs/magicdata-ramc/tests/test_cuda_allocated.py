import torch
from typing import Any
class AllocatedMemContext:
    def __init__(self) -> None:
        # Ensure CUDA libraries are loaded
        torch.cuda.current_blas_handle()
        self.before: dict[str, int] = {}
        self.after: dict[str, int] = {}
        self.delta: dict[str, int] = {}
    def _get_mem_dict(self) -> dict[str,int]:
        # only need `allocated_bytes.all` 
        key_prefix = 'allocated_bytes.all'
        return {k.replace(key_prefix ,''): v for k, v in torch.cuda.memory_stats().items() if key_prefix in k}

    def __enter__(self) -> 'AllocatedMemContext':
        self.before  = self._get_mem_dict()
        return self
    def __exit__(self,*args: Any, **kwargs:Any) -> None:
        self.after = self._get_mem_dict()
        self.delta = {k:v - self.before[k] for k, v in self.after.items()}

with AllocatedMemContext() as mem:
    a = torch.randn(2**8,device=torch.device("cuda")) # 1KiB
    a2 = torch.randn(2**8, device=torch.device("cuda")) # 1 KiB
    del a2
    a3 = torch.randn(2**8, device=torch.device("cuda")) # 1 KiB
    del a3

print(mem.before)
print(mem.after)
print(mem.delta)
