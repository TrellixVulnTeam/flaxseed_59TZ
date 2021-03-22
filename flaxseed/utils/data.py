from abc import ABC
from concurrent.futures import ProcessPoolExecutor
import os
import random
from typing import Iterable, Sequence, Callable, Optional

import jax.numpy as np


class Dataset(ABC):
    def __init__(
        self,
        root: str,
        transform: Callable = None,
        target_transform: Callable = None,
        **kwargs
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.max_workers = 2 ** 20

    def __len__(self):
        """Computes the length of the dataset (total number of samples)."""

    def __getitem__(self, index: int):
        """Retrieves a single sample from the dataset.  Each sample may contain
        multiple entries (e.g. image and label pair).
        """


class DatasetFolder(Dataset):
    """A generic dataset where the samples are arranged in this way:

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    This example uses images, but in general they could be any file type.  User
    must provide a 'loader' function for each sample file.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load an image given its path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        loader: Callable,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert os.path.isdir(root), "Invalid dataset root: '{root}'."
        assert loader is not None, "Must provide a 'loader' function."
        self.loader = loader

        folders = filter(lambda f: os.path.isdir(os.path.join(root, f)), root)
        folder_to_idx = {os.path.join(root, f): i for i, f in enumerate(folders)}
        self.data = [
            (os.path.join(f, file_name), i)
            for f, i in folder_to_idx.items()
            for file_name in os.listdir(f)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        path, y = self.data[index]
        x = self.loader(path)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


class Subset(Dataset):
    def __init__(self, dataset: Dataset, indices: Iterable[int]):
        self._superset = dataset
        self.indices = list(indices)
        self.max_workers = dataset.max_workers

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        return self._superset[self.indices[index]]


class Concatenate(Dataset):
    def __init__(self, datasets: Iterable[Dataset]):
        self._subsets = tuple(datasets)
        self._start_indices = ()
        self._len = -1
        self.max_workers = min(d.max_workers for d in datasets)

    def __len__(self):
        if self._len < 0:
            self._len = sum(len(subset) for subset in self._subsets)
        return self._len

    @property
    def start_indices(self):
        if len(self._start_indices) == 0:
            lengths = [len(subset) for subset in self._subsets]
            self._start_indices = np.cumsum(lengths) - len(self._subsets[0])
        return self._start_indices

    def __getitem__(self, index: int):
        subset_idx = np.nonzero(index >= self.start_indices)[0][-1]
        offset = index - self.start_indices[subset_idx]
        return self._subsets[subset_idx][offset]


class DataLoader(Sequence):
    """
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        max_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        max_workers: int = 0,
        collate_fn: Callable = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self._shuffle = shuffle
        self.drop_last = drop_last
        self.max_workers = min(max_workers, dataset.max_workers)
        self.collate_fn = collate_fn if collate_fn else default_collate

        self._batch_idx = 0
        self._sample_indices = list(range(0, len(dataset)))
        if shuffle:
            self.shuffle()

    def shuffle(self):
        random.shuffle(self._sample_indices)

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size + int(self.drop_last)

    def __iter__(self):
        return self

    def __getitem__(self, idx: int):
        start = idx * self.batch_size
        end = start + self.batch_size

        if self.max_workers > 0:
            pool = ProcessPoolExecutor(max_workers=self.max_workers)
            samples = tuple(
                pool.map(self.dataset.__getitem__, self._sample_indices[start:end])
            )
        else:
            samples = tuple(self.dataset[i] for i in self._sample_indices[start:end])

        return self.collate_fn(samples)

    def __next__(self):
        if self._batch_idx < len(self):
            out = self[self._batch_idx]
            self._batch_idx += 1
            return out
        else:
            self._batch_idx = 0
            if self._shuffle:
                self.shuffle()
            raise StopIteration


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    if isinstance(elem, (np.ndarray, float, int)):
        return np.array(batch)
    elif isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, Sequence):
        return tuple(default_collate(samples) for samples in zip(*batch))
    else:
        raise TypeError(
            f"Data type {type(elem)} not supported by default collate function. "
            f"Supported types: [np.ndarray, float, int, dict]."
        )
