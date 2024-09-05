#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from torch.optim import Optimizer
from typing_extensions import override

import warnings
from abc import ABC, abstractmethod
from typing import TypeAlias
from torch.optim.lr_scheduler import _LRScheduler

LRScheduler: TypeAlias = _LRScheduler


class AbstractLRScheduler(ABC, LRScheduler):
    """Provides a skeletal implementation of :class:`LRScheduler`."""

    @final
    @override
    def get_lr(self) -> list[float]:  # type: ignore[override]
        if not self._get_lr_called_within_step:  # type: ignore[attr-defined]
            warnings.warn(
                "To get the last learning rate computed by the scheduler, use `get_last_lr()`."
            )

        return self._compute_lrs()

    @abstractmethod
    def _compute_lrs(self) -> list[float]:
        """Compute the learning rate of each parameter group."""


def _get_per_param_group(
    optimizer: Optimizer, name: str, value: float | Sequence[float]
) -> Sequence[float]:
    num_param_groups = len(optimizer.param_groups)

    if isinstance(value, float):
        return [value] * num_param_groups

    if len(value) != num_param_groups:
        raise ValueError(
            f"The length of `{name}` must be equal to the number of parameter groups ({num_param_groups}), but is {len(value)} instead."
        )

    return value

@final
class PolynomialDecayLR(AbstractLRScheduler):
    """Represents the polynomial decay learning rate schedule.

    **During warmup:**

    .. math::
        \\eta_t = \\eta_{base} \\frac{t}{T_{warmup}}

    **After warmup:**

    .. math::
        \\eta_t = \\eta_{final} + (\\eta_{base} - \\eta_{final}) (\\frac{T - t}{T - T_{warmup}})^{p}

    This corresponds to increasing the learning rate linearly for the first
    :math:`T_{warmup}` training steps to the base learning rate, and decreasing
    it thereafter for :math:`T - T_{warmup}` steps to the final learning rate
    using a polynomial of degree :math:`p`.

    .. note::
        This scheduler is not chainable.
    """

    _num_steps: int
    _num_warmup_steps: int
    _power: float
    _start_lrs: Sequence[float]
    _final_lrs: Sequence[float]

    def __init__(
        self,
        optimizer: Optimizer,
        num_steps: int,
        num_warmup_steps: int,
        *,
        power: float = 1.0,
        start_lr: float | Sequence[float] = 0.0,
        final_lr: float | Sequence[float] = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        :param optimizer:
            The optimizer to associate.
        :param num_steps:
            The total number of steps, including warmup, over which to decay the
            learning rate.
        :param num_warmup_steps:
            The number of warmup steps.
        :param power:
            The exponent of the polynomial used for decay.
        :param start_lr:
            The initial warmup learning rate of all parameter groups, or of each
            parameter group respectively.
        :param final_lr:
            The final learning rate of all parameter groups, or of each
            parameter group respectively.
        :param last_epoch:
            The index of the last epoch.
        """
        if num_warmup_steps >= num_steps:
            raise ValueError(
                f"`num_warmup_steps` must be less than `num_steps` ({num_steps}), but is {num_warmup_steps} instead."
            )

        self._num_steps = num_steps
        self._num_warmup_steps = num_warmup_steps
        self._power = power

        self._start_lrs = _get_per_param_group(optimizer, "start_lr", start_lr)
        self._final_lrs = _get_per_param_group(optimizer, "final_lr", final_lr)

        super().__init__(optimizer, last_epoch)

    @override
    def _compute_lrs(self) -> list[float]:
        base_lrs = self.base_lrs

        # The decay is already complete, return the final learning rate.
        if self.last_epoch >= self._num_steps:
            return [f for f in self._final_lrs]

        # Linearly increase the learning rate to its base value during warmup.
        if self.last_epoch < self._num_warmup_steps:
            c = self.last_epoch / self._num_warmup_steps

            return [s + (b - s) * c for b, s in zip(base_lrs, self._start_lrs)]

        # After the warmup, decay the learning rate to its final value.
        r = self._num_steps - self.last_epoch
        t = self._num_steps - self._num_warmup_steps

        c = (r / t) ** self._power

        return [f + (b - f) * c for b, f in zip(base_lrs, self._final_lrs)]

