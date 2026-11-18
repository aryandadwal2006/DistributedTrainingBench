FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "bench.train"]
CMD ["--device", "cuda", "--steps", "20", "--warmup", "5"]
