from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _silu_forward_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        result = values / (1.0 + tl.exp(-values))
        tl.store(y_ptr + offsets, result, mask=mask)


def _triton_forward(x: torch.Tensor) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable; install requirements-triton.txt")
    if not x.is_cuda:
        raise RuntimeError("the Triton activation requires a CUDA tensor")
    source = x.contiguous()
    output = torch.empty_like(source)
    block_size = 1024
    _silu_forward_kernel[(triton.cdiv(source.numel(), block_size),)](
        source,
        output,
        source.numel(),
        BLOCK_SIZE=block_size,
    )
    return output


class _TritonSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return _triton_forward(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        sigmoid = torch.sigmoid(x)
        grad = grad_output * sigmoid * (1.0 + x * (1.0 - sigmoid))
        return (grad,)


def activation(x: torch.Tensor, kernel: str) -> torch.Tensor:
    if kernel == "torch":
        return torch.nn.functional.silu(x)
    if kernel == "triton":
        return _TritonSiLU.apply(x)
    raise ValueError(f"unknown activation kernel: {kernel}")
