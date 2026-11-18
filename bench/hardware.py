from __future__ import annotations

import torch


def benchmark_memory(runtime, iterations: int = 20) -> dict:
    if runtime.device.type != "cuda":
        return {"available": False, "reason": "CUDA is required"}
    free_bytes, _ = torch.cuda.mem_get_info(runtime.device)
    properties = torch.cuda.get_device_properties(runtime.device)
    l2_bytes = int(getattr(properties, "L2_cache_size", 0) or 0)
    small_bytes = max(256 * 1024, min(2 * 1024 * 1024, l2_bytes // 2 if l2_bytes else 1024 * 1024))
    large_bytes = min(64 * 1024 * 1024, max(small_bytes * 4, free_bytes // 32))
    large_bytes = max(small_bytes, large_bytes)

    def measure(buffer_bytes: int) -> float:
        element_count = max(1, buffer_bytes // 4)
        source = torch.empty(element_count, device=runtime.device, dtype=torch.float32)
        destination = torch.empty_like(source)
        source.fill_(1.0)
        for _ in range(5):
            destination.copy_(source)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            destination.copy_(source)
        end.record()
        end.synchronize()
        elapsed_seconds = max(start.elapsed_time(end) / 1000.0, 1e-9)
        transferred_bytes = 2 * buffer_bytes * iterations
        return transferred_bytes / elapsed_seconds / 1e9

    small_bandwidth = measure(small_bytes)
    large_bandwidth = measure(large_bytes)
    return {
        "available": True,
        "l2_cache_bytes_reported": l2_bytes,
        "small_buffer_bytes": small_bytes,
        "small_buffer_copy_gb_per_second": small_bandwidth,
        "large_buffer_bytes": large_bytes,
        "large_buffer_copy_gb_per_second": large_bandwidth,
        "iterations": iterations,
        "note": "copy bandwidth is a directional probe, not a hardware-counter measurement",
    }
