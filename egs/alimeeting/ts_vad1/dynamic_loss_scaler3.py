#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast, final
import logging

import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer
import torch.distributed as dist

log = logging.getLogger(__name__)

@final
class DynamicLossScaler:
    """Performs loss scaling during backward pass to prevent underflow of half precision gradients."""

    _optimizer: Optimizer
    _scale_window: int
    _min_scale: float
    _is_enabled: bool
    _grad_scaler: GradScaler | ShardedGradScaler

    def __init__(
        self,
        optimizer: Optimizer,
        world_size: int,
        *,
        sharded: bool = True,
        init_scale: float = 2.0**15,
        scale_factor: float = 2.0,
        scale_window: int | None = None,
        min_scale: float = 0.0,
        gradient_accumulation: int = 1,
        enabled: bool = True,
    ) -> None:
        self._validate_optimizer(optimizer, enabled)
        scale_window = self._compute_scale_window(enabled, world_size, scale_window, gradient_accumulation)

        if not enabled or not sharded or world_size == 1:
            self._grad_scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=scale_factor,
                backoff_factor=1 / scale_factor,
                growth_interval=scale_window,
                enabled=enabled,
            )
        else:
            self._grad_scaler = ShardedGradScaler(
                init_scale=init_scale,
                growth_factor=scale_factor,
                backoff_factor=1 / scale_factor,
                growth_interval=scale_window,
                enabled=enabled,
                process_group=dist.group.WORLD,
            )

        self._optimizer = optimizer
        self._scale_window = scale_window
        self._min_scale = min_scale
        self._is_enabled = enabled

    def _validate_optimizer(self, optimizer: Optimizer, enabled: bool) -> None:
        if enabled:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.dtype not in (torch.float32, torch.float16):
                        raise ValueError(f"Unsupported parameter dtype: {param.dtype}")
                    if param.device.type != "cuda":
                        raise ValueError(f"Parameters must be on CUDA, found on: {param.device.type}")

    def _compute_scale_window(self, enabled: bool, world_size: int, scale_window: int | None, gradient_accumulation: int) -> int:
        if scale_window is None:
            if enabled:
                scale_window = max(int(2**14 / world_size / gradient_accumulation), 1)
                log.info(f"The scale window is set to {scale_window}.")
            else:
                scale_window = 1
        return scale_window

    def run_optimizer_step(self, step_nr: int, closure: Callable[[], float] | None = None) -> tuple[float | None, LossScaleResult]:
        """Perform a single optimization step."""
        loss = self._grad_scaler.step(self._optimizer, closure)
        return loss, self._update_scale(step_nr)

    def _update_scale(self, step_nr: int) -> LossScaleResult:
        old_scale = self._grad_scaler.get_scale()
        self._grad_scaler.update()
        new_scale = self._grad_scaler.get_scale()

        if self._are_close(old_scale, new_scale):
            return LossScaleResult(old_scale, new_scale)

        if new_scale > old_scale:
            log.info(f"No gradient overflow detected in the last {self._scale_window} step(s) after step {step_nr}, increasing loss scale from {old_scale} to {new_scale}.")

            return LossScaleResult(old_scale, new_scale)

        if self._min_scale > new_scale:
            self._grad_scaler.update(self._min_scale)

            if self._are_close(old_scale, self._min_scale):
                log.error(f"Overflow detected at step {step_nr}, ignoring gradient, loss scale is already at minimum ({self._min_scale}). Try lowering the learning rate, using gradient clipping, or increasing the batch size.")
            else:
                log.error(f"Overflow detected at step {step_nr}, ignoring gradient, decreasing loss scale from {old_scale} to {self._min_scale} (minimum). Try lowering the learning rate, using gradient clipping, or increasing the batch size.")

            return LossScaleResult(old_scale, new_scale, overflow=True, min_reached=True)
        else:
            log.info(f"Overflow detected at step {step_nr}, ignoring gradient, decreasing loss scale from {old_scale} to {new_scale}.")
            return LossScaleResult(old_scale, new_scale, overflow=True)

    @staticmethod
    def _are_close(a: float, b: float) -> bool:
        """Check if two floats are close to each other."""
        return math.isclose(a, b, rel_tol=1.3e-6, abs_tol=1e-5)

    def unscale_gradients_(self) -> None:
        self._grad_scaler.unscale_(self._optimizer)
    def state_dict(self) -> dict[str, Any]:
        return {"grad_scaler": self._grad_scaler.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        try:
            self._grad_scaler.load_state_dict(state_dict["grad_scaler"])
        except KeyError as ex:
            raise ValueError(
                "`state_dict` must contain the state of the internal `GradScaler`."
            ) from ex
    # 省略其他方法以简化展示

@final
@dataclass(frozen=True)
class LossScaleResult:
    """Holds the result of a loss scale operation."""
    old_scale: float
    new_scale: float
    overflow: bool = False
    min_reached: bool = False
