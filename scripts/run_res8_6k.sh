#!/bin/bash
# depth 6/3/6, reduced_resolution=8 (64x64), batch=32, 1000 epochs
# checkpointing every 200 epochs = 5600 steps (28 iters/epoch x 200)

PYTHON=${PYTHON:-python}
mkdir -p exps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--model SiT-S/2
  --spa-depth 6 --tem-depth 3 --afno-depth 6 --encoder-depth 4
  --batch-size 32
  --epochs 1000
  --lr-cycles 3
  --reduced-resolution 8
  --sampling-steps 5600
  --checkpointing-steps 5600
  --allow-tf32
  --pretrained-mae-path exps/mae/vivit-M0.1-mask0.5-mae_final.pt
  --base-path data/
  --flnm 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5
  --output-dir exps/
  --logging-dir logs
  --num-workers 4"

echo "Starting FourierFlow (AFNO) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON -u train.py \
  $COMMON \
  --mixer afno \
  --exp-name fourierflow_res8 \
  > exps/run_fourierflow_res8.log 2>&1 &
echo "  PID: $!"

echo "Starting WaveletFlow (db4, level=2) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON -u train.py \
  $COMMON \
  --mixer wavelet --wavelet-wave db4 --wavelet-level 2 \
  --exp-name waveletflow_db4_res8 \
  > exps/run_waveletflow_db4_res8.log 2>&1 &
echo "  PID: $!"

echo "Starting WaveletFlow (haar, level=2) on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u train.py \
  $COMMON \
  --mixer wavelet --wavelet-wave haar --wavelet-level 2 \
  --exp-name waveletflow_haar_res8 \
  > exps/run_waveletflow_haar_res8.log 2>&1 &
echo "  PID: $!"

echo ""
echo "All 3 jobs launched. Monitor with:"
echo "  tail -f exps/run_fourierflow_res8.log"
echo "  tail -f exps/run_waveletflow_db4_res8.log"
echo "  tail -f exps/run_waveletflow_haar_res8.log"
