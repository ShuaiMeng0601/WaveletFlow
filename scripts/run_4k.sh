#!/bin/bash
# Run all 3 models for 4000 epochs on 3 separate GPUs in parallel
# batch_size=64, spa/tem/afno depth=3/2/3
# CyclicLR: 3 cycles auto-computed from total steps (4000*14=56000)
# Checkpointing every 200 epochs = 2800 steps

PYTHON=/data/shuaim/miniconda3/envs/waveletflow/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--model SiT-S/2
  --spa-depth 3 --tem-depth 2 --afno-depth 3 --encoder-depth 3
  --batch-size 32
  --epochs 1000
  --lr-cycles 3
  --sampling-steps 5600
  --checkpointing-steps 5600
  --allow-tf32
  --pretrained-mae-path exps/mae/vivit-M0.1-mask0.5-mae_final.pt
  --base-path data/
  --flnm 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5
  --output-dir exps/
  --logging-dir exps/logs
  --num-workers 4"

echo "Starting FourierFlow (AFNO) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON -u train.py \
  $COMMON \
  --mixer afno \
  --exp-name fourierflow_4k \
  > exps/run_fourierflow_4k.log 2>&1 &
echo "  PID: $!"

echo "Starting WaveletFlow (db4, level=2) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON -u train.py \
  $COMMON \
  --mixer wavelet --wavelet-wave db4 --wavelet-level 2 \
  --exp-name waveletflow_db4_4k \
  > exps/run_waveletflow_db4_4k.log 2>&1 &
echo "  PID: $!"

echo "Starting WaveletFlow (haar, level=2) on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u train.py \
  $COMMON \
  --mixer wavelet --wavelet-wave haar --wavelet-level 2 \
  --exp-name waveletflow_haar_4k \
  > exps/run_waveletflow_haar_4k.log 2>&1 &
echo "  PID: $!"

echo ""
echo "All 3 jobs launched. Monitor with:"
echo "  tail -f exps/run_fourierflow_4k.log"
echo "  tail -f exps/run_waveletflow_db4_4k.log"
echo "  tail -f exps/run_waveletflow_haar_4k.log"
