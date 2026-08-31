# WaveletFlow

**Spatially Localized Wavelet Mixing for Flow Matching in Turbulence Modeling**

WaveletFlow is a research prototype for conditional generation of turbulent
flow trajectories. It replaces the Fourier-mixing branch in
[FourierFlow](https://github.com/AI4Science-WestlakeU/FourierFlow) with a
multi-level wavelet mixer so that frequency processing remains localized in
space.

> This repository is built on FourierFlow. Our main additions are the wavelet
> mixer, the AFNO-Wavelet hybrid option, MAE training support, and comparison
> scripts. The project is currently a research prototype; pretrained
> checkpoints are not included.

## Method

The model receives four flow snapshots and generates the next four snapshots.
The wavelet branch:

1. applies a 2D discrete wavelet transform (DWT) to spatial tokens;
2. processes detail coefficients at each scale with a learnable MLP;
3. applies learnable scale-dependent weighting;
4. reconstructs the feature map with the inverse DWT;
5. fuses the result with the spatial-temporal attention branch.

Three mixer variants share the same training entry point:

- `afno`: the FourierFlow/AFNO baseline;
- `wavelet`: the WaveletFlow branch;
- `hybrid`: AFNO followed by wavelet residual refinement.

## Installation

```bash
git clone https://github.com/ShuaiMeng0601/WaveletFlow.git
cd WaveletFlow

conda create -n waveletflow python=3.9 -y
conda activate waveletflow
pip install -r requirements.txt
```

## Dataset

Download a compressible Navier-Stokes dataset from
[PDEBench](https://github.com/pdebench/PDEBench) and place the HDF5 file under
`data/`, or pass a different directory through `--base-path`/`--data-path`.

The current minimal workflow expects the PDEBench fields `density`, `pressure`,
`Vx`, `Vy`, `x-coordinate`, and `y-coordinate`.

## Train the MAE encoder

```bash
python train_mae.py \
  --data-path data \
  --flnm 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5 \
  --output-dir exps/mae
```

## Train WaveletFlow

```bash
accelerate launch train.py \
  --model SiT-S/2 \
  --mixer wavelet \
  --wavelet-wave db4 \
  --wavelet-level 2 \
  --base-path data \
  --flnm 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5 \
  --pretrained-mae-path exps/mae/vivit-M0.1-mask0.5-mae_final.pt \
  --batch-size 32 \
  --epochs 1000 \
  --checkpointing-steps 5600 \
  --sampling-steps 5600 \
  --output-dir exps \
  --logging-dir logs \
  --allow-tf32
```

Use `--mixer afno` for the baseline or `--mixer hybrid` for the combined
variant. The scripts in `scripts/` launch several single-GPU comparison runs;
set `PYTHON` if your Python executable is not available as `python`.

## Evaluation

Training directories include a timestamp. Pass that full directory name to
`--exp-name`:

```bash
python eval.py \
  --output-dir exps \
  --exp-name waveletflow_0831-12:00 \
  --ckpt-step 5600 \
  --model SiT-S/2 \
  --encoder-depth 4 \
  --base-path data \
  --mixer wavelet \
  --wavelet-wave db4 \
  --wavelet-level 2
```

Evaluation reports RMSE, normalized RMSE, and maximum error, and writes sample
visualizations under `exps/visualizations/`.

`all_eval.py` and `all_eval_step.py` are legacy upstream comparison scripts
that additionally require a separate `FourierFlowSurrogate` checkout. They are
not required for the minimal WaveletFlow workflow above.

## Tests

```bash
pytest -q
```

The smoke tests cover the wavelet block, a small end-to-end model forward pass,
and Euler/Heun sampling. GitHub Actions runs the same checks on CPU.

## Reproducibility notes

- Classifier-free guidance is intentionally disabled (`cfg_scale=1`) because
  the current training pipeline does not train a null-condition branch.
- Dataset splits and training seeds are controlled through `--seed`.
- For a fair AFNO/Wavelet comparison, keep the dataset, training updates,
  optimizer settings, and evaluation sampler identical.

## Acknowledgements

WaveletFlow is based on
[FourierFlow](https://github.com/AI4Science-WestlakeU/FourierFlow), which in
turn builds on [DiT](https://github.com/facebookresearch/DiT) and
[SiT](https://github.com/willisma/SiT). Wavelet transforms are provided by
[pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets).

## License

Released under the MIT License. See `LICENSE`.
