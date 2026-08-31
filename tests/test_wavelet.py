import torch

from models.wno2d import WNO2D


def test_wavelet_block_preserves_shape_and_backpropagates():
    torch.manual_seed(0)
    model = WNO2D(hidden_size=8, level=1, wave="haar", mlp_ratio=2.0)
    inputs = torch.randn(2, 8 * 8, 8, requires_grad=True)

    outputs = model(inputs)
    outputs.mean().backward()

    assert outputs.shape == inputs.shape
    assert inputs.grad is not None
    assert torch.isfinite(outputs).all()
