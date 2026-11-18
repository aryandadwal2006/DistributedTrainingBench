from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import torch
import transformers


def hardware_metadata(device: torch.device) -> dict:
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        try:
            nccl_version = torch.cuda.nccl.version()
        except Exception:
            nccl_version = None
        metadata.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": properties.total_memory,
                "gpu_multiprocessors": properties.multi_processor_count,
                "gpu_l2_cache_bytes": getattr(properties, "L2_cache_size", None),
                "nccl": nccl_version,
            }
        )
    else:
        metadata.update(
            {
                "gpu_name": None,
                "gpu_total_memory_bytes": None,
                "gpu_multiprocessors": None,
                "gpu_l2_cache_bytes": None,
                "nccl": None,
            }
        )
    return metadata


def build_record(
    args,
    runtime,
    elapsed: float,
    loss: float | None,
    parameter_count: int,
    peak_allocated: int | None = None,
    peak_reserved: int | None = None,
) -> dict:
    data_parallel_factor = runtime.world_size if args.parallel in {"dp", "fsdp"} else 1
    token_count = args.batch_size * args.seq_len * args.steps * data_parallel_factor
    record = {
        "task": args.task,
        "parallel": args.parallel,
        "world_size": runtime.world_size,
        "kernel": args.kernel,
        "compiled": args.compile,
        "compile_mode": args.compile_mode if args.compile else "eager",
        "dtype": args.dtype,
        "checkpointing": args.checkpointing,
        "batch_size_per_rank": args.batch_size,
        "sequence_length": args.seq_len,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "heads": args.heads,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "elapsed_seconds": elapsed,
        "average_step_ms": elapsed * 1000.0 / args.steps,
        "global_tokens_per_second": token_count / elapsed if elapsed > 0 else 0.0,
        "loss": loss,
        "total_parameters": parameter_count,
        "peak_allocated_bytes_on_rank": peak_allocated or 0,
        "peak_reserved_bytes_on_rank": peak_reserved or 0,
        "memory_hierarchy": None,
        "seed": args.seed,
    }
    record.update(hardware_metadata(runtime.device))
    return record


def write_record(output_dir: str, record: dict) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "results.csv"
    fields = list(record.keys())
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    print(json.dumps(record, indent=2, default=str), flush=True)
