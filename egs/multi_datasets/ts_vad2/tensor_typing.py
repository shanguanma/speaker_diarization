# it is from https://github.com/lucidrains/transfusion-pytorch/blob/main/transfusion_pytorch/tensor_typing.py

"""
how to use this ?
global ein notation

b - batch
n - sequence
d - dimension
l - logits (text)

from tensor_typing import Float,Int,Bool
e.g.
x: Float["b n d"]
def ()-> Float['b n l'] | Float['']:
Float[""] means that a float number
| means that or
"""

from torch import Tensor

from jaxtyping import (
    Float,
    Int,
    Bool
)

# jaxtyping is a misnomer, works for pytorch

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(Float)
Int   = TorchTyping(Int)
Bool  = TorchTyping(Bool)

__all__ = [
    Float,
    Int,
    Bool
]
