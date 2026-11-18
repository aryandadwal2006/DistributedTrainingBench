# DistributedTrainingBench

A small, reproducible benchmark for compact Transformer training and accelerator behavior. It uses synthetic token batches and a local Hugging Face `PreTrainedModel` configuration, so it does not download checkpoints or datasets and does not require credentials.

## What it measures

- Training and inference: compact causal Transformer workloads, eager PyTorch, `torch.compile` / Inductor, optional Triton SiLU, mixed precision, and gradient checkpointing.
- Parallelism: single process, DistributedDataParallel (DDP), FullyShardedDataParallel (FSDP), tensor parallel MLPs, and a minimal GPipe-style pipeline split.
- Communication: blocking versus asynchronous NCCL/Gloo all-reduce with independent matrix-multiply work between launch and wait.
- Hardware: step time, global tokens/second, parameter count, CUDA allocated/reserved/peak memory, device properties, an optional small/large-buffer copy-bandwidth probe, and runtime versions.
- Profiling: PyTorch profiler Chrome traces and NVTX ranges for Nsight Systems.
- Deployment: Docker, Kubernetes Job, and OpenShift Job examples.

The benchmark intentionally stays small. It does not claim to implement production-grade fault tolerance, multi-node rendezvous management, or a full 1F1B pipeline scheduler.

## Requirements

- Python 3.10+
- PyTorch 2.4+
- Transformers 4.45+
- CUDA-capable NVIDIA GPU for CUDA, NCCL, FSDP, and Triton runs
- Linux is recommended for Triton and multi-GPU execution

CPU execution is supported for the basic model, eager training, and unit tests. DDP can use Gloo on CPU. FSDP requires CUDA in this implementation. Tensor and pipeline parallel runs require at least two distributed processes.

## Install

### Linux / CUDA

Install a PyTorch wheel matching the host NVIDIA driver and CUDA runtime using the [official PyTorch install selector](https://pytorch.org/get-started/locally/). Then:

```bash
git clone https://github.com/aryandadwal2006/DistributedTrainingBench.git
cd DistributedTrainingBench
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Triton kernels on supported Linux/CUDA systems:

```bash
python -m pip install -r requirements-triton.txt
```

### Windows

The eager, compiled, and CPU smoke-test paths can be used on Windows. Multi-GPU NCCL and Triton are intended for Linux. Use the PyTorch installation command appropriate for your Windows/CUDA setup before installing the remaining requirements.

## Quick start

A CPU smoke test:

```bash
python -m bench.train --device cpu --steps 5 --warmup 1 --batch-size 2 --seq-len 32 --hidden-size 64 --layers 2 --heads 4
```

A single-GPU training run:

```bash
python -m bench.train --device cuda --steps 50 --warmup 10 --dtype bf16
```

A single-GPU inference throughput run:

```bash
python -m bench.train --task infer --device cuda --steps 100 --warmup 10 --dtype fp16
```

Inference uses locally initialized synthetic model weights and synthetic tokens. It measures forward-pass throughput and latency, not task quality or pretrained-model performance. No checkpoint download is performed.

Results are appended to `outputs/results.csv`. A JSON summary is printed by rank zero. Each run records its parallel mode, kernel, compiler setting, precision, throughput, latency, memory, and device metadata.

## Compare eager, Inductor, and Triton

Run the same compact workload in three configurations:

```bash
bash scripts/compare.sh
```

The script runs eager PyTorch, `torch.compile` / Inductor, and the Triton activation. It expects one CUDA device. Edit the `DEVICE`, `STEPS`, and `WARMUP` values at the top of the script to adjust the workload. Triton runs require `requirements-triton.txt`.

Equivalent individual runs:

```bash
python -m bench.train --device cuda --kernel torch
python -m bench.train --device cuda --compile --kernel torch
python -m bench.train --device cuda --kernel triton
```

Compilation is warmed up before measured steps. Compare runs on an otherwise idle device with identical model and workload settings. The first compiled iteration is not included in the reported steady-state timing.

## Inference benchmarking

Inference is a separate forward-only path. It supports eager execution, `torch.compile` / Inductor, FP16/BF16 autocast, Triton activation, profiler traces, and the optional CUDA memory probe. It runs as a single process per GPU replica:

```bash
python -m bench.train --task infer --device cuda --batch-size 8 --seq-len 128 --steps 100 --warmup 10
```

The result CSV distinguishes `train` and `infer` runs. Inference uses random local weights unless the implementation is extended with a checkpoint loader; do not interpret its output as model-quality evaluation.

## Distributed training

Launch one process per GPU on a single node:

```bash
torchrun --standalone --nproc_per_node=2 -m bench.train --device cuda --parallel dp --steps 50 --dtype bf16
```

Select a mode with `--parallel`:

| Mode | Implementation | Notes |
| --- | --- | --- |
| `none` | Single process | Default |
| `dp` | PyTorch DDP | Each rank receives a different synthetic batch |
| `fsdp` | PyTorch FSDP | CUDA only; uses full-shard strategy |
| `tp` | DTensor tensor parallel | Shards each MLP's first linear column-wise and second linear row-wise |
| `pp` | Manual autograd send/recv pipeline | Contiguous layer partitions; simple synchronous GPipe-style schedule |

For two-GPU FSDP:

```bash
torchrun --standalone --nproc_per_node=2 -m bench.train --device cuda --parallel fsdp --steps 50
```

For two-GPU tensor parallelism:

```bash
torchrun --standalone --nproc_per_node=2 -m bench.train --device cuda --parallel tp --steps 50
```

For two-stage pipeline parallelism:

```bash
torchrun --standalone --nproc_per_node=2 -m bench.train --device cuda --parallel pp --steps 50
```

Distributed mode requires `torchrun` and `WORLD_SIZE >= 2`. The launcher initializes NCCL for CUDA and Gloo for CPU. For multi-node runs, provide the usual `torchrun` rendezvous options (`--nnodes`, `--node_rank`, `--master_addr`, and `--master_port`) and ensure all nodes can reach the rendezvous host.

Tensor parallelism currently targets the two MLP projections in each Transformer block; attention, embeddings, and the LM head are replicated. Pipeline parallelism is a minimal correctness-oriented schedule, not an overlap-optimized production pipeline. Do not combine TP, PP, or FSDP with `--compile` in this minimal benchmark. DDP plus `--compile` is supported.

## Mixed precision and memory

Supported precision values are `fp32`, `fp16`, and `bf16`. FP16 uses a CUDA gradient scaler. BF16 uses autocast without scaling. Some devices do not support BF16; use FP16 or FP32 on those devices.

Enable gradient checkpointing with:

```bash
python -m bench.train --device cuda --checkpointing --dtype bf16
```

The model dimensions, batch size, sequence length, warmup count, and measured step count are CLI options. Increase dimensions gradually to avoid out-of-memory errors.

Run an optional small/large-buffer CUDA copy probe to compare a cache-sized working set with a larger allocation:

```bash
python -m bench.train --device cuda --memory-benchmark
```

The result records the reported L2 cache size and effective copy bandwidth for both buffer sizes. This is a lightweight probe, not a substitute for hardware performance counters.

## Communication-computation overlap

On a distributed run, compare blocking all-reduce with asynchronous all-reduce while a matrix multiplication executes before `work.wait()`:

```bash
torchrun --standalone --nproc_per_node=2 -m bench.train --device cuda --parallel dp --comm-benchmark --comm-mb 16
```

The communication benchmark is skipped for single-process runs. It reports average blocking and asynchronous-overlap round times. The matrix multiplication is independent work intended to expose whether communication can progress concurrently on the selected backend and hardware.

## Profiling and Nsight

Export a PyTorch profiler trace:

```bash
python -m bench.train --device cuda --profile --profile-steps 5
```

The Chrome trace is written to `outputs/profile-rank0.json`. The training loop also emits NVTX ranges on CUDA, so it can be captured with Nsight Systems:

```bash
nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --output=outputs/nsys-report python -m bench.train --device cuda --steps 20 --warmup 5
nsys stats outputs/nsys-report.nsys-rep
```

Run `nsys` from a host with Nsight Systems installed. GPU performance counters and some memory-hierarchy details are hardware- and driver-dependent; the benchmark records properties exposed by PyTorch and does not attempt privileged counter collection.

## Docker

Build and run on a host with the NVIDIA Container Toolkit:

```bash
docker build -t distributed-training-bench:local .
docker run --rm --gpus all distributed-training-bench:local --device cuda --steps 20 --warmup 5
docker run --rm --gpus all distributed-training-bench:local --task infer --device cuda --steps 100 --warmup 10
```

Run distributed workers in the container:

```bash
docker run --rm --gpus all --ipc=host --entrypoint torchrun distributed-training-bench:local --standalone --nproc_per_node=2 -m bench.train --device cuda --parallel dp --steps 20
```

For multi-process distributed execution, invoke the image through `torchrun` (see the Kubernetes manifests) or override the container entrypoint.

## Kubernetes

Edit the GPU resource count and image in `deploy/kubernetes-job.yaml`, then:

```bash
kubectl apply -f deploy/kubernetes-job.yaml
kubectl logs -f job/distributed-training-bench
```

The training manifest requests two NVIDIA GPUs and launches two local DDP workers. The inference manifest requests one GPU and runs a forward-only workload. The cluster must have the NVIDIA device plugin or GPU Operator installed, and the image must be available to the cluster.

## OpenShift

Edit the image in `deploy/openshift-job.yaml`, then:

```bash
oc apply -f deploy/openshift-job.yaml
oc logs -f job/distributed-training-bench
```

The training manifest requests two NVIDIA GPUs for DDP; the inference manifest requests one GPU for a forward-only run. The OpenShift project must have permission to request the `nvidia.com/gpu` resource and pull the image. Adapt the service account and image-pull policy to the cluster's security policies.

## CLI

```text
--task {train,infer}
--device {auto,cpu,cuda}
--parallel {none,dp,fsdp,tp,pp}
--kernel {torch,triton}
--compile
--compile-mode {default,reduce-overhead,max-autotune}
--dtype {fp32,fp16,bf16}
--checkpointing
--batch-size N
--seq-len N
--vocab-size N
--hidden-size N
--layers N
--heads N
--steps N
--warmup N
--lr FLOAT
--seed N
--output-dir PATH
--profile
--profile-steps N
--memory-benchmark
--comm-benchmark
--comm-mb FLOAT
```

## Tests

```bash
python -m pip install pytest
pytest -q
```

Tests use a tiny CPU model and do not require CUDA, datasets, model downloads, or API keys.

## Reproducibility notes

- Workloads use synthetic token IDs generated from a seeded per-rank generator.
- The model is initialized locally from a Hugging Face configuration.
- Warmup steps are excluded from steady-state timing.
- The output CSV stores the main benchmark settings alongside measured results.
- For defensible comparisons, keep the model shape, batch, sequence length, precision, warmup, step count, software versions, and device fixed.
- Throughput is calculated from global batch tokens divided by the slowest rank's measured duration.

## Scope

This project is a compact benchmark harness, not a training framework. It does not download pretrained weights, provide a tokenizer or task-quality metric, implement elastic recovery, or guarantee performance gains from compilation, Triton, or asynchronous collectives. Results are measurements from the machine and software configuration on which the benchmark is run.
