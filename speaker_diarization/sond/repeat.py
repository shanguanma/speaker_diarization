#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn
import torch

class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args


def repeat(N, fn):
    """Repeat module N times.

    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.

    Returns:
        MultiSequential: Repeated model instance.

    """
    return MultiSequential(*[fn(n) for n in range(N)])
