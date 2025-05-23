# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import timedelta
from enum import Enum
from typing import Any, final

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import Backend, ProcessGroup, ReduceOp
from typing_extensions import override

from device import determine_default_cuda_device, determine_default_device
from logging_me import get_log_writer
from typing_me import CPU, Device
from env import get_int_from_env

log = get_log_writer(__name__)

def setup_root_gang(
    log: LogWriter,
    *,
    timeout: timedelta | None = None,
    monitored: bool = False,
) -> Gang:
    """Set up the root gang.

    :param log:
        The log to write to.
    :param timeout:
        The timeout for collective operations.
    :param monitored:
        If ``True``,  puts a monitored barrier before every collective call.
    """
    device = determine_default_device()

    #log_environment_info(log, device)

    # In case we run on Ampere or later, use TF32.
    torch.set_float32_matmul_precision("high")

    log.info("Initializing the root gang.")

    gang = setup_default_gang(timeout=timeout, monitored=monitored)

    log.info("Root gang initialized.")

    return gang

class ReduceOperation(Enum):
    """Specifies a reduce operation."""

    SUM = 1
    MEAN = 2
    PRODUCT = 3
    MIN = 4
    MAX = 5


class Gang(ABC):
    """Represents a set of processes that work collectively."""

    @abstractmethod
    def close(self) -> None:
        """Close and destroy the gang."""

    @abstractmethod
    def create_gang(self, ranks: Sequence[int]) -> Gang | None:
        """Create a new gang.

        :param ranks:
            The ranks of processes that will be part of the new gang.
        """

    @abstractmethod
    def as_process_group(self) -> ProcessGroup:
        """Return this gang as a process group."""

    @abstractmethod
    def barrier(self) -> None:
        """Synchronize all processes."""

    @abstractmethod
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        """Reduce ``tensor`` across all processes.

        :param tensor:
            The input and output tensor of the operation.
        :param op:
            The element-wise reduce operation.
        """

    @abstractmethod
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        """Gather tensors from all processes and put them in ``output_tensor``.

        :param output_tensor:
            The output tensor to accomodate tensors from all processes.
        :param input_tensor:
            The tensor to be gathered from this process.
        """

    @abstractmethod
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None:
        """Gather tensors from all processes and put them in ``output_tensors``.

        :param output_tensors:
            The tensor list to accomodate tensors from all processes.
        :param input_tensor:
            The tensor to be gathered from this process.
        """

    @abstractmethod
    def broadcast(self, tensor: Tensor, source_rank: int = 0) -> None:
        """Broadcast ``tensor`` from ``source_rank`` to all processes.

        :param tensor:
            The tensor to be sent from ``source_rank``.
        :param source_rank:
            The rank of the process from which to broadcast ``tensor``.
        """

    @abstractmethod
    def broadcast_objects(self, objects: list[Any], source_rank: int = 0) -> None:
        """Broadcast picklable ``objects`` from ``source_rank`` to all processes.

        :param objects:
            The list of picklable objects to broadcast. Each process must
            provide lists of equal sizes.
        :param source_rank:
            The rank of the process from which to broadcast ``objects``.
        """

    @property
    @abstractmethod
    def rank(self) -> int:
        """The rank of this process in the gang."""

    @property
    @abstractmethod
    def size(self) -> int:
        """The number of processes that are part of the gang."""

    @property
    @abstractmethod
    def device(self) -> Device:
        """The associated device."""


class AbstractGang(Gang):
    """Provides a skeletal implementation of :class:`Gang`."""

    _rank: int
    _size: int
    _device: Device

    def __init__(self, rank: int, size: int, device: Device) -> None:
        """
        :param rank:
            The rank of this process in the gang.
        :param size:
            The number of processes that are part of the gang.
        :param device:
            The associated device.
        """
        if size == 0:
            raise ValueError("`size` must be greater than zero.")

        if rank >= size:
            raise ValueError(
                f"`rank` must be less than `size` ({size}), but is {rank} instead."
            )

        self._rank = rank
        self._size = size

        self._device = device

    @final
    @override
    def create_gang(self, ranks: Sequence[int]) -> Gang | None:
        if len(set(ranks)) != len(ranks):
            raise ValueError("The ranks in ``ranks`` must be all unique.")

        for idx, rank in enumerate(ranks):
            if rank < 0 or rank > self._size:
                raise ValueError(
                    f"The rank at index {idx} in ``ranks`` must be greater than or equal to 0 and less than the size of the gang ({self._size}), but is {rank} instead."
                )

        return self._do_create_gang(ranks)

    @abstractmethod
    def _do_create_gang(self, ranks: Sequence[int]) -> Gang | None:
        """Create a new gang.

        :param ranks:
            The ranks of processes that will be part of the new gang.
        """

    @final
    @property
    @override
    def rank(self) -> int:
        return self._rank

    @final
    @property
    @override
    def size(self) -> int:
        return self._size

    @final
    @property
    @override
    def device(self) -> Device:
        return self._device


@final
class FakeGang(AbstractGang):
    """Represents a non-distributed gang for local use."""

    def __init__(
        self, *, rank: int = 0, size: int = 1, device: Device | None = None
    ) -> None:
        """
        :param rank:
            The emulated rank of this process in the gang.
        :param size:
            The emulated number of processes that are part of the gang.
        :param device:
            If ``None``; if CUDA is available, the gang will use the default
            CUDA device of the process; otherwise, it will use the CPU.
        """
        if device is None:
            device = determine_default_device()

        super().__init__(rank=rank, size=size, device=device)

    @override
    def close(self) -> None:
        pass

    @override
    def _do_create_gang(self, ranks: Sequence[int]) -> FakeGang | None:
        try:
            idx = ranks.index(self._rank)
        except ValueError:
            return None

        return FakeGang(rank=idx, size=len(ranks), device=self._device)

    @override
    def as_process_group(self) -> ProcessGroup:
        raise RuntimeError("`FakeGang` does not support conversion to a process group.")

    @override
    def barrier(self) -> None:
        pass

    @override
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        match op:
            case ReduceOperation.SUM:
                tensor *= self._size
            case ReduceOperation.PRODUCT:
                tensor.pow_(self._size)
            case _:
                raise ValueError(
                    "`FakeGang` supports only `SUM` and `PRODUCT` reduce operations."
                )

    @override
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        if not output_tensor.is_contiguous():
            raise ValueError("`output_tensor` must be contiguous.")

        if output_tensor.dim() != input_tensor.dim() + 1:
            raise ValueError(
                "`output_tensor` must have a shape that is compatible with all-gather."
            )

        if output_tensor.size(0) != self._size:
            raise ValueError(
                f"The size of the first dimension of `output_tensor` must match the size of the gang ({self._size}), but is {output_tensor.size(0)} instead."
            )

        for i in range(self._size):
            output_tensor[i].copy_(input_tensor)

    @override
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None:
        if len(output_tensors) != self._size:
            raise ValueError(
                f"The length of `output_tensors` must match the size of the gang ({self._size}), but is {len(output_tensors)} instead."
            )

        for i in range(self._size):
            output_tensors[i].copy_(input_tensor)

    @override
    def broadcast(self, tensor: Tensor, source_rank: int = 0) -> None:
        if source_rank != self._rank:
            raise ValueError(
                f"`source_rank` must be {self._rank}, but is {source_rank} instead."
            )

    @override
    def broadcast_objects(self, objects: list[Any], source_rank: int = 0) -> None:
        if source_rank != self._rank:
            raise ValueError(
                f"`source_rank` must be {self._rank}, but is {source_rank} instead."
            )


@final
class ProcessGroupGang(AbstractGang):
    """Represents a gang that wraps a process group."""

    _default: ProcessGroupGang | None = None

    _pg: ProcessGroup
    _monitor_pg: ProcessGroup | None

    def __init__(
        self,
        pg: ProcessGroup,
        device: Device,
        *,
        monitor_pg: ProcessGroup | None = None,
    ) -> None:
        super().__init__(dist.get_rank(pg), dist.get_world_size(pg), device)

        self._pg = pg
        self._monitor_pg = monitor_pg

    @classmethod
    def init_default_process_group(
        cls,
        *,
        device: Device | None = None,
        timeout: timedelta | None = None,
        num_threads: int | None = None,
        monitored: bool = False,
        ok_initialized: bool = False,
    ) -> ProcessGroupGang:
        """Initialize the default process group and wrap it as a gang.

        :param device:
            If ``None``; if CUDA is available, the gang will use the default
            CUDA device of the process; otherwise, it will use the CPU.
        :param timeout:
            The timeout for collective operations. If ``None``, the default
            timeout value (15 minutes) will be used.
        :param num_threads:
            The number of threads to use for interaop parallelism.
        :param monitored:
            If ``True``,  puts a monitored barrier before every collective call.
        :param ok_initialized:
            If ``True``, does not raise an error if the default process group is
            already initialized.
        """
        if log.is_enabled_for_debug():
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

            dist.set_debug_level_from_env()

        if not dist.is_available():
            raise RuntimeError("`torch.distributed` is not available.")

        if dist.is_initialized():
            log.info(f"dist.is_initialized(): {dist.is_initialized()}")
            if ok_initialized:
                log.info("Default process group is already initialized. Skipping initialization.")  # fmt: skip

                return ProcessGroupGang.from_default_process_group()

            raise RuntimeError("The default process group is already initialized.")

        num_procs = get_local_world_size()

        if num_threads is None:
            if num_procs > 1 and "OMP_NUM_THREADS" not in os.environ:
                # To prevent thread oversubscription, we distribute cores evenly
                # across the workers.
                num_threads = _get_num_cpus(num_procs)

        if num_threads is not None:
            torch.set_num_threads(num_threads)

            log.info("Setting the number of threads used for intraop parallelism to {}.", num_threads)  # fmt: skip

        if device is None:
            device = determine_default_device()

            assert device.type == "cpu" or device.type == "cuda"

        backend: str | None

        if device.type == "cpu":
            backend = Backend.GLOO
        elif device.type == "cuda":
            backend = Backend.NCCL
        else:
            raise ValueError(
                f"`device` must be of type `cpu` and `cuda`, but is of type `{device.type}` instead."
            )

        if device.type == "cuda":
            # See https://github.com/pytorch/pytorch/issues/46874.
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

        if timeout is None:
            timeout = timedelta(minutes=15)

        dist.init_process_group(backend, timeout=timeout)

        pg = dist.group.WORLD
        if pg is None:
            raise RuntimeError(
                "The default process group is not available. Please file a bug report."
            )

        if monitored:
            if backend == Backend.GLOO:
                monitor_pg = pg
            else:
                # Gloo is needed for monitored barrier support.
                monitor_pg = dist.new_group(backend=Backend.GLOO, timeout=timeout)
        else:
            monitor_pg = None

        cls._default = ProcessGroupGang(pg, device, monitor_pg=monitor_pg)

        return cls._default

    @staticmethod
    def from_process_group(pg: ProcessGroup, device: Device) -> ProcessGroupGang:
        """Wrap ``pg`` as a gang.

        :param pg:
            The process group to wrap.
        :param device:
            The associated device.
        """
        return ProcessGroupGang(pg, device)

    @classmethod
    def from_default_process_group(cls) -> ProcessGroupGang:
        """Wrap the default process group as a gang."""
        if not dist.is_available():
            raise RuntimeError("`torch.distributed` is not available.")

        if not dist.is_initialized():
            raise RuntimeError("The default process group is not initialized.")

        if cls._default is not None:
            return cls._default

        backend = dist.get_backend()

        match backend:
            case Backend.GLOO:
                device = CPU
            case Backend.NCCL:
                cuda_device = determine_default_cuda_device()
                if cuda_device is None:
                    raise RuntimeError(
                        "The default process group uses the `nccl` backend, but the `cuda` device cannot be determined. Please file a bug report."
                    )

                device = cuda_device
            case _:
                raise RuntimeError(
                    f"Only `nccl` and `gloo` backends are supported, but the process group uses the `{backend}` backend."
                )

        if dist.group.WORLD is None:
            raise RuntimeError(
                "The default process group is not available. Please file a bug report."
            )

        cls._default = ProcessGroupGang(dist.group.WORLD, device)

        return cls._default

    @override
    def close(self) -> None:
        dist.destroy_process_group(self._pg)

    @override
    def _do_create_gang(self, ranks: Sequence[int]) -> ProcessGroupGang | None:
        if self._pg is not dist.group.WORLD:
            raise RuntimeError(
                "`create_gang()` can only be called on the gang associated with the default (i.e. main) process group."
            )

        backend = dist.get_backend()

        pg = dist.new_group(ranks, backend=backend)

        if self._rank not in ranks:
            return None

        if self._monitor_pg is not None:
            if backend == Backend.GLOO:
                monitor_pg = pg
            else:
                monitor_pg = dist.new_group(ranks, backend=Backend.GLOO)
        else:
            monitor_pg = None

        return ProcessGroupGang(pg, self._device, monitor_pg=monitor_pg)

    @override
    def as_process_group(self) -> ProcessGroup:
        return self._pg

    @override
    def barrier(self) -> None:
        if self._monitor_pg is None:
            dist.barrier(group=self._pg, device_ids=[self._device.index])
        else:
            torch.cuda.synchronize()

            dist.monitored_barrier(group=self._monitor_pg, wait_all_ranks=True)

    @override
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        self._maybe_monitored_barrier()

        dist.all_reduce(tensor, self._get_reduce_op(op), group=self._pg)

    @override
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        self._maybe_monitored_barrier()

        dist.all_gather_into_tensor(output_tensor, input_tensor, group=self._pg)

    @override
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None:
        self._maybe_monitored_barrier()

        dist.all_gather(output_tensors, input_tensor, group=self._pg)

    @override
    def broadcast(self, tensor: Tensor, source_rank: int = 0) -> None:
        self._maybe_monitored_barrier()

        dist.broadcast(tensor, source_rank, group=self._pg)

    @override
    def broadcast_objects(self, objects: list[Any], source_rank: int = 0) -> None:
        self._maybe_monitored_barrier()

        dist.broadcast_object_list(objects, source_rank, group=self._pg)

    def _maybe_monitored_barrier(self) -> None:
        if self._monitor_pg is None:
            return

        torch.cuda.synchronize()

        dist.monitored_barrier(group=self._monitor_pg, wait_all_ranks=True)

    @staticmethod
    def _get_reduce_op(op: ReduceOperation):  # type: ignore[no-untyped-def]
        if op == ReduceOperation.SUM:
            return ReduceOp.SUM
        if op == ReduceOperation.MEAN:
            return ReduceOp.AVG  # type: ignore[attr-defined]
        if op == ReduceOperation.PRODUCT:
            return ReduceOp.PRODUCT
        if op == ReduceOperation.MIN:
            return ReduceOp.MIN
        if op == ReduceOperation.MAX:
            return ReduceOp.MAX

        raise ValueError(
            f"`op` must be an operation supported by the underlying process group, but is `{op}` instead."
        )


def _get_num_cpus(num_procs: int) -> int:
    num_cpus = os.cpu_count()

    affinity_mask = os.sched_getaffinity(0)

    if num_cpus is None or affinity_mask is None:
        log.warning("The number of CPUs cannot be determined.")

        return 1

    # We should not exceed the number of cores available in the affinity mask.
    return min(max(num_cpus // num_procs, 1), len(affinity_mask))


def get_world_size() -> int:
    """Return the world size of the running job."""
    #value = get_int_from_env("WORLD_SIZE")
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1

    #return 1 if value is None else value


def get_rank() -> int:
    """Return the rank of this process in the running job."""
    value = get_int_from_env("RANK", allow_zero=True)

    return 0 if value is None else value


def get_local_world_size() -> int:
    """Return the local world size of the running job."""
    value = get_int_from_env("LOCAL_WORLD_SIZE")

    return 1 if value is None else value


def get_local_rank() -> int:
    """Return the local rank of this process in the running job."""
    value = get_int_from_env("LOCAL_RANK", allow_zero=True)

    return 0 if value is None else value


def setup_default_gang(
    *,
    device: Device | None = None,
    timeout: timedelta | None = None,
    monitored: bool = False,
) -> Gang:
    """Set up the default gang of this process.

    :param device:
        If ``None``; if CUDA is available, the gang will use the default CUDA
        device of the process; otherwise, it will use the CPU.
    :param timeout:
        The timeout for collective operations.
    :param monitored:
        If ``True``,  puts a monitored barrier before every collective call.
    """
    if get_world_size() == 1:
        return FakeGang(device=device)
    else:
        return ProcessGroupGang.init_default_process_group(
        device=device, timeout=timeout, monitored=monitored, ok_initialized=True
    )


def all_sum(gang: Gang, value: float | int | Tensor) -> Tensor:
    """Sum ``value`` over all processes in ``gang``."""
    if isinstance(value, Tensor):
        output = value
    else:
        output = torch.tensor(value, device=gang.device)

    gang.all_reduce(output, ReduceOperation.SUM)

    return output
