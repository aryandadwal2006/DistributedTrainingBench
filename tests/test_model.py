import torch

from bench.model import BenchmarkConfig, DistributedTrainingModel


def make_model(checkpointing=False):
    config = BenchmarkConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        gradient_checkpointing=checkpointing,
    )
    return DistributedTrainingModel(config)


def test_forward_shape_and_loss():
    model = make_model()
    tokens = torch.randint(0, 64, (2, 16))
    loss, logits = model(tokens, labels=tokens)
    assert logits.shape == (2, 16, 64)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_backward_with_gradient_checkpointing():
    model = make_model(checkpointing=True)
    model.train()
    tokens = torch.randint(0, 64, (2, 16))
    loss, _ = model(tokens, labels=tokens)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_inference_without_labels():
    model = make_model()
    loss, logits = model(torch.randint(0, 64, (1, 8)))
    assert loss is None
    assert logits.shape == (1, 8, 64)
