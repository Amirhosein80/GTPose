import random
import numpy as np
import torch


class RandomResample:
    def __init__(self, target_fps_list, p=0.5):
        self.p = p
        self.target_fps_list = target_fps_list

    def __call__(self, samples, list_keys):
        if random.random() < self.p:
            for key in list_keys:
                target_fps = np.random.choice(self.target_fps_list)
                indices = torch.arange(0, samples[key].shape[1], 60 / target_fps)

                start_indices = torch.floor(indices).long()
                end_indices = torch.ceil(indices).long()
                end_indices[end_indices >= samples[key].shape[1]] = samples[key].shape[1] - 1  # handling edge cases

                start = samples[key][:, start_indices]
                end = samples[key][:, end_indices]

                floats = indices - start_indices
                floats = floats.unsqueeze(0)
                for shape_index in range(len(samples[key].shape) - 2):
                    floats = floats.unsqueeze(-1)
                weights = torch.ones_like(start) * floats
                samples[key] = torch.lerp(start, end, weights)
            
        return samples


class UniformNoise:
    def __init__(self, noise=0.1, p=0.5):
        self.p = p
        self.noise = noise

    def __call__(self, samples, list_keys):
        for key in list_keys:
            if random.random() < self.p:
                noise = torch.zeros_like(samples[key]).uniform_(-self.noise, self.noise)
                samples[key] += noise
        return samples
