import codecs
import gzip
import os
import pickle
from typing import Callable, Optional

import numpy as np

from flaxseed.utils.data import Dataset
from flaxseed.utils.download import download_url


__all__ = ["MNIST", "FashionMNIST", "KMNIST"]


class MNIST(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where data files exist.
        train (bool, optional): If True, returns the training set.  Else, the test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    resources = [
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    ]
    classes = [str(i) for i in range(10)]

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.folder = os.path.join(self.root, "MNIST")
        self.training_file = os.path.join(self.folder, "training.pkl.gz")
        self.test_file = os.path.join(self.folder, "test.pkl.gz")
        self.max_workers = 0

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        path = self.training_file if train else self.test_file
        with open(path, "rb") as f:
            self.data, self.targets = pickle.load(f)

    def __getitem__(self, index: int):
        x, y = self.data[index], int(self.targets[index])
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def __len__(self) -> int:
        return len(self.data)

    def _check_exists(self) -> bool:
        return os.path.exists(self.training_file) and os.path.exists(self.test_file)

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.folder, exist_ok=True)

        for url, md5 in self.resources:
            filename = url.rpartition("/")[2]
            download_url(url, root=self.folder, filename=filename, md5=md5)

        print("Processing...")
        training_set = (
            load_images(os.path.join(self.folder, "train-images-idx3-ubyte.gz")),
            load_labels(os.path.join(self.folder, "train-labels-idx1-ubyte.gz")),
        )
        test_set = (
            load_images(os.path.join(self.folder, "t10k-images-idx3-ubyte.gz")),
            load_labels(os.path.join(self.folder, "t10k-labels-idx1-ubyte.gz")),
        )
        with open(self.training_file, "wb") as f:
            pickle.dump(training_set, f)
        with open(self.test_file, "wb") as f:
            pickle.dump(test_set, f)

        print("Done!")


class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where data files exist.
        train (bool, optional): If True, returns the training set.  Else, the test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    resources = [
        (
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        ),
        (
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "25c81989df183df01b3e8a0aad5dffbe",
        ),
        (
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "bef4ecab320f06d8554ea6380940ec79",
        ),
        (
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
            "bb300cfdad3c16e7a12a480ee83cd310",
        ),
    ]
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]


class KMNIST(MNIST):
    """`Kuzushiji-MNIST <https://github.com/rois-codh/kmnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where data files exist.
        train (bool, optional): If True, returns the training set.  Else, the test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    resources = [
        (
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz",
            "bdb82020997e1d708af4cf47b453dcf7",
        ),
        (
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz",
            "e144d726b3acfaa3e44228e80efcd344",
        ),
        (
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz",
            "5c965bf0a639b31b8f53240b1b52f4d7",
        ),
        (
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz",
            "7320c461ea6c1c855c0b718fb2a4b134",
        ),
    ]
    classes = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz'.
    """
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    else:
        return open(path, "rb")


def read_ubyte(path):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """

    with open_maybe_compressed_file(path) as f:
        data = f.read()

    def get_int(b):
        return int(codecs.encode(b, "hex"), 16)

    magic = get_int(data[0:4])
    ndim = magic % 256
    size = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(ndim)]
    parsed = np.frombuffer(data, dtype=np.uint8, offset=(4 * (ndim + 1)))

    return parsed.astype(np.uint8, copy=False).reshape(*size)


def load_labels(path):
    return read_ubyte(path).astype(np.int)


def load_images(path):
    return read_ubyte(path).astype(np.float) / 255
