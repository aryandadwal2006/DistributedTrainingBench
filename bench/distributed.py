from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass
class Runtime:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    distributed: bool


def initialize_runtime(requested_device: str) -> Runtime:
    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    elif torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    else:
        device = torch.device("cpu")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if device.type == "cuda":
        torch.cuda.set_device(device)

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    return Runtime(
        device=device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=world_size > 1,
    )


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_model(model: torch.nn.Module, parallel: str, runtime: Runtime) -> torch.nn.Module:
    if parallel == "none":
        if runtime.distributed:
            raise RuntimeError("use --parallel dp, fsdp, tp, or pp with torchrun")
        return model

    if not runtime.distributed:
        raise RuntimeError(f"--parallel {parallel} requires torchrun with at least two processes")

    if parallel == "dp":
        device_ids = [runtime.local_rank] if runtime.device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=device_ids)

    if parallel == "fsdp":
        if runtime.device.type != "cuda":
            raise RuntimeError("FSDP mode requires CUDA in this benchmark")
        from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

        return FullyShardedDataParallel(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=runtime.device,
            use_orig_params=True,
        )

    if parallel == "tp":
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Replicate, Shard
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            parallelize_module,
        )

        mesh = init_device_mesh(runtime.device.type, (runtime.world_size,))
        plan = {}
        for index in range(len(model.layers)):
            plan[f"layers.{index}.mlp.fc1"] = ColwiseParallel(
                output_layouts=Shard(-1),
                use_local_output=False,
            )
            plan[f"layers.{index}.mlp.fc2"] = RowwiseParallel(
                input_layouts=Shard(-1),
                output_layouts=Replicate(),
            )
        return parallelize_module(model, mesh, plan)

    if parallel == "pp":
        raise RuntimeError("pipeline parallelism is handled by bench.pipeline")

    raise ValueError(f"unknown parallel mode: {parallel}")


def distributed_max(value: float, runtime: Runtime) -> float:
    if not runtime.distributed:
        return value
    tensor = torch.tensor(value, device=runtime.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def cleanup_runtime() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
