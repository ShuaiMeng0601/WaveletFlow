import torch
import h5py
import os
import numpy as np
import sys

# Ensure we can import from parent directory
sys.path.append(os.getcwd())
from data.CNS_data_utils import FNODatasetMultistep

def save_test_subset():
    filename = "2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5"
    saved_folder = "data/"
    output_filename = "data/2D_CFD_Test_Subset_8GB.h5"
    
    if os.path.exists(output_filename):
        print(f"File {output_filename} already exists. Skipping.")
        return

    print(f"Loading 83GB data and splitting (this will take ~8 mins)...")
    # Use the exact same parameters as training/eval to get the same split
    train_dataset, test_dataset, normalizer = FNODatasetMultistep.get_train_test_datasets(
        filename,
        reduced_resolution=4,
        reduced_resolution_t=1,
        reduced_batch=1,
        initial_step=0,
        saved_folder=saved_folder,
        if_eval_plot=True
    )
    
    print(f"Extraction complete. Test samples: {len(test_dataset.data)}")
    
    # test_dataset.data is [B, H, W, T, C] (Normalized)
    # We will save the RAW data so the file remains a valid CFD dataset
    raw_test_data = normalizer.decode(test_dataset.data.numpy())
    
    print(f"Saving to {output_filename}...")
    with h5py.File(output_filename, 'w') as f:
        # Structure expected by FNODatasetMultistep: [batch, time, x, y]
        # raw_test_data is [B, H, W, T, C] -> Transpose to [B, T, H, W, C]
        raw_test_data = np.transpose(raw_test_data, (0, 3, 1, 2, 4))
        
        f.create_dataset('density', data=raw_test_data[..., 0])
        f.create_dataset('pressure', data=raw_test_data[..., 1])
        f.create_dataset('Vx', data=raw_test_data[..., 2])
        f.create_dataset('Vy', data=raw_test_data[..., 3])
        
        # Grid (128x128)
        grid = test_dataset.grid.numpy()
        f.create_dataset('x-coordinate', data=grid[:, 0, 0])
        f.create_dataset('y-coordinate', data=grid[0, :, 1])
        f.create_dataset('t-coordinate', data=np.arange(raw_test_data.shape[1]))
        
        # CRITICAL: Save normalizer statistics so eval.py doesn't need train_dataset
        f.create_dataset('norm_mean', data=normalizer.mean)
        f.create_dataset('norm_std', data=normalizer.std)

    print(f"Success! Test subset saved to {output_filename}")

if __name__ == "__main__":
    save_test_subset()
