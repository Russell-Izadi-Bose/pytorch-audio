"""
Russell Izadi
2022
"""

import torch
from torch.utils.data import Dataset


class Transform(Dataset):
    """
    Transform a dataset
    """
    def __init__(
            self,
            dataset,
            transform,
            index: int = None,
            dtype=torch.float32) -> None:
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.index = index
        self.dtype = dtype

    def __getitem__(self, i: int):
        sample = self.dataset[i]
        if isinstance(sample, tuple):
            assert self.index < len(sample)
            sample = list(sample)
            sample[self.index] = self.transform(sample[self.index])
            sample = tuple(sample)
        else:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.dataset)
