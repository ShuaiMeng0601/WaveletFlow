#!/bin/bash
# Run all 3 models for 3000 epochs on 3 separate GPUs in parallel
# batch_size=128, spa/tem/afno depth=3/2/3, checkpointing every 200 epochs (1400 steps)

PYTHON=${PYTHON:-python}
mkdir -p exps
COMMON="--model SiT-S/2
  --spa-depth 3 --tem-depth 2 --afno-depth 3
  --batch-size 128
  --epochs 3000
  --sampling-steps 1400
  --checkpointing-steps 1400
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
  --exp-name fourierflow_3k \
  > exps/run_fourierflow_3k.log 2>&1 &
echo "  PID: $!"

echo "Starting WaveletFlow (db4, level=2) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON -u train.py \
  $COMMON \
  --mixer wavelet --wavelet-wave db4 --wavelet-level 2 \
  --exp-name waveletflow_db4_3k \
  > exps/run_waveletflow_db4_3k.log 2>&1 &
echo "  PID: $!"

echo "Starting WaveletFlow (haar, level=2) on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u train.py \
  $COMMON \
  --mixer wavelet --wavelet-wave haar --wavelet-level 2 \
  --exp-name waveletflow_haar_3k \
  > exps/run_waveletflow_haar_3k.log 2>&1 &
echo "  PID: $!"

echo ""
echo "All 3 jobs launched. Monitor with:"
echo "  tail -f exps/run_fourierflow_3k.log"
echo "  tail -f exps/run_waveletflow_db4_3k.log"
echo "  tail -f exps/run_waveletflow_haar_3k.log"
