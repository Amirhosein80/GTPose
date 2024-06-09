import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from utils import collate_fn, gen_mean_std


class CustomDataset(Dataset):
    def __init__(self, root, phase, transform=None, seq_length=256):
        assert phase in ["Train", "Valid", "All"], "You should select phase between Train and Valid and All (both Train & Valid)"
        self.files = glob.glob(os.path.join(root, f"{phase}_seq", f"seq_{seq_length}", "*.npz"))
        self.phase = phase
        self.transform = transform
        self.normalize_keys = ["imu_acc", "javel", "jlacc", "jvel"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = dict(np.load(file))

        for key, value in data.items():
            if key != "dataset":
                data[key] = torch.from_numpy(value)
                
        if self.transform is not None:
            data = self.transform(data, self.normalize_keys)
        return data

    def get_data_loader(self, batch_size, num_workers):
        if self.phase in ["Train", "All"]:
            sampler = RandomSampler(self)
        else:
            sampler = SequentialSampler(self)
        return DataLoader(self, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)
    
    def gen_mean_std(self):
        return gen_mean_std(dl=dl, keys=self.normalize_keys,
                       path="../data", phase=self.phase)


if __name__ == '__main__':
    ds = CustomDataset(root="../data", phase="Train")
    dl = ds.get_data_loader(1024, 4)

