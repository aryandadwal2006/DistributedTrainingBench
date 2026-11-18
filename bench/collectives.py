from __future__ import annotations

import time

import torch
import torch.distributed as dist


def benchmark_collectives(runtime, size_mb: float, iterations: int = 20, warmup: int = 5) -> dict | None:
    if not runtime.distributed:
        return None
    if size_mb <= 0:
        raise ValueError("--comm-mb must be positive")

    element_count = max(1, int(size_mb * 1024 * 1024 / 4))
    communication = torch.ones(element_count, device=runtime.device, dtype=torch.float32)
    left = torch.randn(512, 512, device=runtime.device)
    right = torch.randn(512, 512, device=runtime.device)

    def compute():
        torch.mm(left, right)

    def synchronize():
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)

    for _ in range(warmup):
        dist.all_reduce(communication)
        work = dist.all_reduce(communication, async_op=True)
        compute()
        work.wait()
    synchronize()
    dist.barrier()

    start = time.perf_counter()
    for _ in range(iterations):
        dist.all_reduce(communication)
        compute()
    synchronize()
    blocking_seconds = time.perf_counter() - start

    dist.barrier()
    start = time.perf_counter()
    for _ in range(iterations):
        work = dist.all_reduce(communication, async_op=True)
        compute()
        work.wait()
    synchronize()
    overlap_seconds = time.perf_counter() - start

    result = {
        "message_size_mb": size_mb,
        "iterations": iterations,
        "blocking_compute_round_ms": blocking_seconds * 1000.0 / iterations,
        "async_overlap_round_ms": overlap_seconds * 1000.0 / iterations,
        "overlap_backend": dist.get_backend(),
    }
    if runtime.rank == 0:
        print({"collective_benchmark": result}, flush=True)
    return result
