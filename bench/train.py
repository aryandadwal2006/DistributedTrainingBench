from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from bench.collectives import benchmark_collectives
from bench.distributed import (
    cleanup_runtime,
    distributed_max,
    initialize_runtime,
    seed_everything,
    wrap_model,
)
from bench.hardware import benchmark_memory
from bench.kernels import TRITON_AVAILABLE
from bench.metrics import build_record, write_record
from bench.model import BenchmarkConfig, DistributedTrainingModel
from bench.pipeline import train_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="distributed compact-Transformer benchmark")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--parallel", choices=("none", "dp", "fsdp", "tp", "pp"), default="none")
    parser.add_argument("--kernel", choices=("torch", "triton"), default="torch")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--memory-benchmark", action="store_true")
    parser.add_argument("--comm-benchmark", action="store_true")
    parser.add_argument("--comm-mb", type=float, default=8.0)
    args = parser.parse_args()

    if args.batch_size < 1 or args.seq_len < 2:
        parser.error("--batch-size must be positive and --seq-len must be at least 2")
    if args.steps < 1 or args.warmup < 0:
        parser.error("--steps must be positive and --warmup cannot be negative")
    if args.layers < 1 or args.hidden_size < 1 or args.heads < 1:
        parser.error("model dimensions must be positive")
    if args.hidden_size % args.heads != 0:
        parser.error("--hidden-size must be divisible by --heads")
    if args.vocab_size < 2:
        parser.error("--vocab-size must be at least 2")
    if args.profile_steps < 1:
        parser.error("--profile-steps must be positive")
    if args.compile and args.parallel in {"fsdp", "tp", "pp"}:
        parser.error("--compile is supported only with parallel=none or parallel=dp")
    if args.compile and args.kernel == "triton":
        parser.error("--compile with --kernel triton is not supported; benchmark these paths separately")
    if args.parallel == "tp" and args.kernel == "triton":
        parser.error("the Triton activation is not supported with tensor parallel DTensors")
    return args


def make_config(args):
    return BenchmarkConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        num_heads=args.heads,
        intermediate_size=4 * args.hidden_size,
        max_position_embeddings=max(args.seq_len, 2),
        gradient_checkpointing=args.checkpointing,
    )


def make_autocast(runtime, precision):
    if precision == "fp32":
        return torch.autocast(device_type=runtime.device.type, enabled=False)
    if runtime.device.type != "cuda":
        raise RuntimeError("fp16 and bf16 autocast require CUDA in this benchmark")
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support bf16")
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_standard(args, runtime, config) -> dict:
    if args.kernel == "triton":
        if runtime.device.type != "cuda":
            raise RuntimeError("the Triton activation requires --device cuda")
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is unavailable; install requirements-triton.txt")

    if args.parallel == "tp" and config.intermediate_size % runtime.world_size != 0:
        raise RuntimeError("tensor parallel world size must divide the MLP intermediate size")

    seed_everything(args.seed)
    model = DistributedTrainingModel(config, kernel=args.kernel).to(runtime.device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model = wrap_model(model, args.parallel, runtime)

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch version")
        model = torch.compile(model, backend="inductor", mode=args.compile_mode)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(runtime.device.type == "cuda" and args.dtype == "fp16"),
    )

    generator = torch.Generator(device=runtime.device)
    data_seed = args.seed + runtime.rank if args.parallel in {"dp", "fsdp"} else args.seed
    generator.manual_seed(data_seed)

    def train_step():
        tokens = torch.randint(
            0,
            args.vocab_size,
            (args.batch_size, args.seq_len),
            generator=generator,
            device=runtime.device,
        )
        optimizer.zero_grad(set_to_none=True)
        if runtime.device.type == "cuda":
            torch.cuda.nvtx.range_push("training_step")
        try:
            with torch.autograd.profiler.record_function("training_step"):
                with make_autocast(runtime, args.dtype):
                    loss, _ = model(tokens, labels=tokens)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        finally:
            if runtime.device.type == "cuda":
                torch.cuda.nvtx.range_pop()
        return loss.detach()

    for _ in range(args.warmup):
        train_step()

    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
        torch.cuda.reset_peak_memory_stats(runtime.device)
    if runtime.distributed:
        dist.barrier()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if runtime.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    last_loss = torch.zeros((), device=runtime.device)
    start = time.perf_counter()
    if args.profile:
        trace_path = Path(args.output_dir) / f"profile-rank{runtime.rank}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        schedule = torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=min(args.profile_steps, args.steps),
            repeat=1,
        )
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=lambda profiler: profiler.export_chrome_trace(str(trace_path)),
        ) as profiler:
            for _ in range(args.steps):
                last_loss = train_step()
                profiler.step()
    else:
        for _ in range(args.steps):
            last_loss = train_step()

    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    elapsed = time.perf_counter() - start
    elapsed = distributed_max(elapsed, runtime)

    if runtime.distributed:
        dist.all_reduce(last_loss, op=dist.ReduceOp.SUM)
        last_loss /= runtime.world_size
    loss_value = float(last_loss.item())

    peak_allocated = (
        torch.cuda.max_memory_allocated(runtime.device)
        if runtime.device.type == "cuda"
        else 0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(runtime.device)
        if runtime.device.type == "cuda"
        else 0
    )
    peak_allocated = int(distributed_max(float(peak_allocated), runtime))
    peak_reserved = int(distributed_max(float(peak_reserved), runtime))

    if runtime.rank != 0:
        return {}

    record = build_record(
        args=args,
        runtime=runtime,
        elapsed=elapsed,
        loss=loss_value,
        parameter_count=parameter_count,
        peak_allocated=peak_allocated,
        peak_reserved=peak_reserved,
    )
    if args.memory_benchmark:
        record["memory_hierarchy"] = benchmark_memory(runtime)
    write_record(args.output_dir, record)
    return record


def main():
    args = parse_args()
    runtime = initialize_runtime(args.device)
    try:
        if args.parallel == "none" and runtime.distributed:
            raise RuntimeError("torchrun detected multiple processes; select a distributed --parallel mode")
        if args.parallel != "none" and not runtime.distributed:
            raise RuntimeError(f"--parallel {args.parallel} requires torchrun with at least two processes")
        if args.parallel == "pp":
            config = make_config(args)
            record = train_pipeline(args, runtime, config)
        else:
            config = make_config(args)
            record = train_standard(args, runtime, config)

        if args.comm_benchmark:
            benchmark_collectives(runtime, args.comm_mb)
        if runtime.rank == 0 and record:
            print(json.dumps({"result_file": str(Path(args.output_dir) / "results.csv")}))
    finally:
        cleanup_runtime()


if __name__ == "__main__":
    main()
