import torch

from models.diff_afno_sit import SiT


def test_small_waveletflow_forward():
    torch.manual_seed(0)
    model = SiT(
        input_size=16,
        patch_size=8,
        in_channels=4,
        hidden_size=32,
        decoder_hidden_size=32,
        num_heads=4,
        afno_depth=1,
        encoder_depth=1,
        spa_depth=1,
        tem_depth=1,
        z_dims=[16],
        projector_dim=32,
        num_frames=4,
        mixer="wavelet",
        wavelet_level=1,
        wavelet_wave="haar",
        fused_attn=False,
        qk_norm=False,
    )
    inputs = torch.randn(2, 4, 4, 16, 16)
    timesteps = torch.tensor([0.25, 0.75])

    outputs, projected_features = model(
        inputs,
        timesteps,
        condition=inputs.clone(),
    )

    assert outputs.shape == inputs.shape
    assert len(projected_features) == 1
    assert torch.isfinite(outputs).all()
