from typing import Callable, Optional

from PIL import Image

from flaxseed.utils.data import DatasetFolder


def default_loader(path: str) -> Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class ImageFolder(DatasetFolder):
    """A generic dataset where the images are arranged in this way:

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Callable] = None,
    ):
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader if loader is not None else default_loader,
        )
