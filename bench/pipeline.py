from __future__ import annotations

import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bench.distributed import seed_everything
from bench.hardware import benchmark_memory
from bench.metrics import build_record, write_record


class _SendActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, destination: int) -> torch.Tensor:
        ctx.destination = destination
        dist.send(tensor.contiguous(), dst=destination)
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = torch.empty_like(grad_output)
        dist.recv(grad_input, src=ctx.destination)
        return grad_input, None


class _ReceiveActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shape, dtype, device, source: int):
        ctx.source = source
        result = torch.empty(shape, dtype=dtype, device=device)
        dist.recv(result, src=source)
        return result.requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dist.send(grad_output.contiguous(), dst=ctx.source)
        return None, None, None, None


class PipelineStage(nn.Module):
    def __init__(self, model, rank: int, world_size: int):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.config = model.config
        layer_count = len(model.layers)
        start = layer_count * rank // world_size
        end = layer_count * (rank + 1) // world_size
        self.layers = nn.ModuleList(list(model.layers[start:end]))
        if rank == 0:
            self.token_embedding = model.token_embedding
            self.position_embedding = model.position_embedding
        if rank == world_size - 1:
            self.final_norm = model.final_norm
            self.lm_head = model.lm_head

    def forward_stage(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length = tokens.shape
        if self.rank == 0:
            positions = torch.arange(length, device=tokens.device).unsqueeze(0)
            hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        else:
            hidden = _ReceiveActivation.apply(
                (batch, length, self.config.hidden_size),
                torch.float32,
                tokens.device,
                self.rank - 1,
            )
        for layer in self.layers:
            hidden = layer(hidden)
        if self.rank < self.world_size - 1:
            hidden = _SendActivation.apply(hidden, self.rank + 1)
        else:
            hidden = self.lm_head(self.final_norm(hidden))
        return hidden


def train_pipeline(args, runtime, config) -> dict:
    if args.dtype != "fp32":
        raise RuntimeError("pipeline mode currently uses fp32 activations; pass --dtype fp32")
    if args.compile:
        raise RuntimeError("pipeline mode cannot be combined with --compile")
    if args.profile:
        raise RuntimeError("use Nsight Systems for pipeline profiling; --profile is for standard training")
    if args.kernel == "triton":
        if runtime.device.type != "cuda":
            raise RuntimeError("the Triton activation requires --device cuda")
        from bench.kernels import TRITON_AVAILABLE

        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is unavailable; install requirements-triton.txt")

    from bench.model import DistributedTrainingModel

    seed_everything(args.seed)
    full_model = DistributedTrainingModel(config, kernel=args.kernel).to(runtime.device)
    parameter_count = sum(parameter.numel() for parameter in full_model.parameters())
    stage = PipelineStage(full_model, runtime.rank, runtime.world_size).to(runtime.device)
    del full_model

    optimizer = torch.optim.AdamW(stage.parameters(), lr=args.lr)
    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(args.seed)
    total_steps = args.warmup + args.steps
    measured_start = None
    last_loss = torch.zeros((), device=runtime.device)

    if args.warmup == 0:
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
            torch.cuda.reset_peak_memory_stats(runtime.device)
        dist.barrier()
        measured_start = time.perf_counter()

    for step in range(total_steps):
        tokens = torch.randint(
            0,
            config.vocab_size,
            (args.batch_size, args.seq_len),
            generator=generator,
            device=runtime.device,
        )
        optimizer.zero_grad(set_to_none=True)
        if runtime.device.type == "cuda":
            torch.cuda.nvtx.range_push("pipeline_step")
        try:
            with torch.autograd.profiler.record_function("pipeline_forward"):
                output = stage.forward_stage(tokens)
                if runtime.rank == runtime.world_size - 1:
                    loss = F.cross_entropy(
                        output[:, :-1, :].contiguous().view(-1, config.vocab_size),
                        tokens[:, 1:].contiguous().view(-1),
                    )
                    last_loss = loss.detach()
                    backward_loss = loss
                else:
                    backward_loss = output.sum() * 0.0
            backward_loss.backward()
            optimizer.step()
        finally:
            if runtime.device.type == "cuda":
                torch.cuda.nvtx.range_pop()

        if args.warmup > 0 and step + 1 == args.warmup:
            if runtime.device.type == "cuda":
                torch.cuda.synchronize(runtime.device)
                torch.cuda.reset_peak_memory_stats(runtime.device)
            dist.barrier()
            measured_start = time.perf_counter()

    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    elapsed = time.perf_counter() - measured_start
    elapsed_tensor = torch.tensor(elapsed, device=runtime.device, dtype=torch.float64)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    elapsed_seconds = float(elapsed_tensor.item())

    dist.all_reduce(last_loss, op=dist.ReduceOp.SUM)
    average_loss = float(last_loss.item())
    allocated = torch.cuda.max_memory_allocated(runtime.device) if runtime.device.type == "cuda" else 0
    reserved = torch.cuda.max_memory_reserved(runtime.device) if runtime.device.type == "cuda" else 0
    allocated_tensor = torch.tensor(float(allocated), device=runtime.device)
    reserved_tensor = torch.tensor(float(reserved), device=runtime.device)
    dist.all_reduce(allocated_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(reserved_tensor, op=dist.ReduceOp.MAX)

    if runtime.rank != 0:
        return {}

    record = build_record(
        args=args,
        runtime=runtime,
        elapsed=elapsed_seconds,
        loss=average_loss,
        parameter_count=parameter_count,
        peak_allocated=int(allocated_tensor.item()),
        peak_reserved=int(reserved_tensor.item()),
    )
    if args.memory_benchmark:
        record["memory_hierarchy"] = benchmark_memory(runtime)
    write_record(args.output_dir, record)
    return record
