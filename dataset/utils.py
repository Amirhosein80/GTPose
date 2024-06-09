import os
import tqdm
import torch

def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for i, k in enumerate(keys):
        if k == "dataset":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.cat([b[k].unsqueeze(0) for b in batch], dim=0)
    return out


def gen_mean_std(dl, keys, path, phase):
    file = os.path.join(path, f'{phase}_mean_std.pt')
    if not os.path.exists(file):
        tars = {}
        for key in keys:
            total_sum = 0
            squared_sum = 0
            num_datapoints = len(dl.dataset)
            loop = tqdm.tqdm(dl, total=len(dl))
            for batch in loop:
                total_sum += torch.sum(batch[key], dim=(0, 1))
                squared_sum += torch.sum(batch[key] ** 2, dim=(0, 1))
                loop.set_description(f"Key: {key}")
            del batch
            mean = total_sum / num_datapoints
            variance = (squared_sum / num_datapoints) - (mean ** 2)
            print(mean, variance)
            std = torch.sqrt(variance)
            tars[key] = {'mean': mean, 'std': std}
        torch.save(tars, os.path.join(path, f'{phase}_mean_std.pt'))
    else: 
        tars = torch.load(file)

    return tars
