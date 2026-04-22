#!/bin/bash
# HybridFlow: AFNO + WNO residual refinement
# depth 6/3/6, reduced_resolution=8 (64x64), batch=32, 1000 epochs

PYTHON=/data/shuaim/miniconda3/envs/waveletflow/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting HybridFlow (AFNO + WNO residual) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON -u train.py \
  --model SiT-S/2 \
  --spa-depth 6 --tem-depth 3 --afno-depth 6 --encoder-depth 4 \
  --batch-size 32 \
  --epochs 1000 \
  --lr-cycles 3 \
  --reduced-resolution 8 \
  --sampling-steps 5600 \
  --checkpointing-steps 5600 \
  --allow-tf32 \
  --mixer hybrid --wavelet-wave db4 --wavelet-level 2 \
  --pretrained-mae-path exps/mae/vivit-M0.1-mask0.5-mae_final.pt \
  --base-path data/ \
  --flnm 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5 \
  --output-dir exps/ \
  --logging-dir exps/logs \
  --num-workers 4 \
  --exp-name hybridflow_res8 \
  > exps/run_hybridflow_res8.log 2>&1 &
echo "  PID: $!"
echo ""
echo "Monitor: tail -f exps/run_hybridflow_res8.log"
