import pytest
import torch

from samplers import euler_sampler


class ZeroVelocity(torch.nn.Module):
    def forward(self, inputs, timesteps, condition, **kwargs):
        assert condition is not None
        return torch.zeros_like(inputs), []


@pytest.mark.parametrize("heun", [False, True])
def test_sampler_preserves_input_for_zero_velocity(heun):
    latents = torch.randn(2, 4, 4, 8, 8)
    condition = torch.randn_like(latents)

    outputs = euler_sampler(
        ZeroVelocity(),
        latents,
        condition,
        num_steps=3,
        heun=heun,
        cfg_scale=1.0,
    )

    assert torch.allclose(outputs, latents.to(torch.float64))


def test_sampler_rejects_untrained_cfg():
    latents = torch.randn(1, 4, 4, 8, 8)

    with pytest.raises(ValueError, match="cfg_scale=1.0"):
        euler_sampler(
            ZeroVelocity(),
            latents,
            latents,
            num_steps=2,
            cfg_scale=2.0,
        )
