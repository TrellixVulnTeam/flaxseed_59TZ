import os
import pickle

import numpy as np

from flaxseed.utils.data import Dataset
from flaxseed.utils.download import download_and_extract_archive, check_integrity


class CIFAR10(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is True.
        train (bool, optional): If True, loads training set.  Else, test set.
        download (bool, optional): If true, downloads the dataset if it does
            not already exist in the `root` directory.
    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]
    test_list = [["test_batch", "40351d587109b95175f43aff81a1287e"]]

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.folder = os.path.join(root, self.__class__.__name__)
        file_list = self.train_list if self.train else self.test_list
        if download:
            self.download()

        self.data = []
        self.targets = []

        for file_name, checksum in file_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                labels_key = "labels" if "labels" in entry else "fine_labels"
                self.data.extend(entry["data"])
                self.targets.extend(entry[labels_key])

        self.data = np.stack(self.data, axis=0)
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self):
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if not self._check_integrity():
            download_and_extract_archive(
                self.url, self.root, filename=self.filename, md5=self.tgz_md5
            )


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset."""
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [["train", "16019d7e3df5f24257cddd939b257f8d"]]
    test_list = [["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"]]
