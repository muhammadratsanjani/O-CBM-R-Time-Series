import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class FI2010Dataset(Dataset):
    def __init__(self, data_list, seq_length=100, k=0):
        self.seq_length = seq_length
        self.k = k
        self.data_list = data_list # list of numpy arrays (N_i, 149)
        
        self.lengths = [max(0, d.shape[0] - seq_length + 1) for d in data_list]
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_samples = self.cumulative_lengths[-1] if len(self.cumulative_lengths) > 0 else 0
        
    def __len__(self):
        return self.total_samples
        
    def __getitem__(self, idx):
        # Find which file this idx belongs to
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[file_idx - 1]
            
        data = self.data_list[file_idx]
        
        # Define window boundaries
        start = local_idx
        end = local_idx + self.seq_length
        
        # Extract raw features
        X_window = data[start:end, 0:40].astype(np.float32)
        
        # Label at the end of window (cast to int to prevent PyTorch TypeError)
        Y = int(data[end-1, 144 + self.k] - 1.0)
        
        # Concepts at the end of window
        C = data[end-1, 40:144].astype(np.float32)
        
        return torch.tensor(X_window), torch.tensor(Y, dtype=torch.long), torch.tensor(C)

def get_lob_dataloader(base_dir="fi2010_data/extracted/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre", batch_size=64, seq_length=100, train=True, max_samples=None):
    data_list = []
    if train:
        file_dir = os.path.join(base_dir, "NoAuction_DecPre_Training")
    else:
        file_dir = os.path.join(base_dir, "NoAuction_DecPre_Testing")
        
    print(f"Searching for txt files in {file_dir}...")
    files = sorted([f for f in os.listdir(file_dir) if f.endswith('.txt')])
    
    for f in files:
        filepath = os.path.join(file_dir, f)
        npy_filepath = filepath.replace('.txt', '.npy')
        
        if not os.path.exists(npy_filepath):
            print(f"Mengonversi {filepath} ke .npy secara berurutan (RAM ultra rendah)...")
            # File FI-2010 memiliki 149 baris, dan ratusan ribu kolom.
            # Pandas read_csv hancur karena mencoba membuat 400.000+ objek kolom.
            # Solusi mutlak: baca per baris menggunakan numpy dari string!
            lines_data = []
            with open(filepath, 'r') as f_in:
                for line in f_in:
                    row = np.fromstring(line, dtype=np.float32, sep=' ')
                    lines_data.append(row)
            
            data = np.stack(lines_data) # (149, N)
            np.save(npy_filepath, data.T) # Simpan dalam transposisi (N, 149)
            del lines_data, data
            import gc
            gc.collect()
            
        print(f"Loading {npy_filepath} via mmap (ZERO RAM) ...")
        # mmap_mode='r' membiarkan OS mengatur RAM. Dataset 10GB pun tidak akan OOM!
        data = np.load(npy_filepath, mmap_mode='r')
        
        if max_samples is not None and data.shape[0] > max_samples:
             data = data[:max_samples, :]
             
        data_list.append(data)
        
    dataset = FI2010Dataset(data_list, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)
    return dataloader

if __name__ == "__main__":
    loader = get_lob_dataloader(train=False, batch_size=32, max_samples=5000)
    for X_batch, Y_batch, C_batch in loader:
        print(f"X shape: {X_batch.shape}")
        print(f"Y shape: {Y_batch.shape}")
        print(f"C shape: {C_batch.shape}")
        break
