# @author Tom Nuno Wolf, Technical University of Munich
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
#
# This file contains both original work and code adapted from:
# - TrainTransform and EvalTransform classes adapted from https://github.com/B-cos/B-cos-v2
# - Original B-cos code: Copyright 2023 Moritz Böhle, Max-Planck-Gesellschaft
# - Remainder of code: Original work by Tom Nuno Wolf

import logging
from pathlib import Path
import copy
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets as tvdatasets
from torchvision import transforms
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode


LOG = logging.getLogger(__name__)


class AddInverse(torch.nn.Module):
    """Converts a tensor of shape [B, C, H, W] input to [B, 2C, H, W]
    by adding the inverse channels of the given one to it.
    """

    def __init__(self, dim: int = -3):
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor: Tensor) -> Tensor:
        return torch.cat([in_tensor, 1 - in_tensor], dim=self.dim)
    

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_INVERSE_NORMALIZE = transforms.Normalize(
        mean=[-mean / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1 / std for std in IMAGENET_STD],
    )

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CIFAR_INVERSE_NORMALIZE = transforms.Normalize(
        mean=[-mean / std for mean, std in zip(CIFAR_MEAN, CIFAR_STD)],
        std=[1 / std for std in CIFAR_STD],
    )

# TrainTransform class adapted from https://github.com/B-cos/B-cos-v2
# Original B-cos code: Copyright 2023 Moritz Böhle, Max-Planck-Gesellschaft
class TrainTransform:
    def __init__(
        self,
        *,
        crop_size=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        random_erase_prob=0.5,
        is_bcos=False,
    ):
        trans = [
                transforms.RandomResizedCrop(crop_size, interpolation=interpolation),
                transforms.RandomHorizontalFlip(hflip_prob),]
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )
        if not is_bcos:
            trans.append(transforms.Normalize(mean=mean, std=std))
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        if is_bcos:
            trans.append(AddInverse())

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

# EvalTransform class adapted from https://github.com/B-cos/B-cos-v2
# Original B-cos code: Copyright 2023 Moritz Böhle, Max-Planck-Gesellschaft
class EvalTransform:
    def __init__(
        self,
        *,
        crop_size=224,
        resize_size=256,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation=InterpolationMode.BILINEAR,
        is_bcos=False,
    ):
        trans = [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            AddInverse()
            if is_bcos
            else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


def get_imagenet_dataloaders(data_dir, batch_size, num_workers, is_bcos=True):
    data_dir = Path(data_dir)
    generator = torch.Generator().manual_seed(42)

    train_data = tvdatasets.ImageFolder(
        data_dir / "train",
        transform=TrainTransform(is_bcos=is_bcos),
    )
    eval_data = tvdatasets.ImageFolder(
        data_dir / "val",
        transform=EvalTransform(is_bcos=is_bcos),
    )
    support_data = tvdatasets.ImageFolder(
        data_dir / "train",
        transform=EvalTransform(is_bcos=is_bcos),
    )
    train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            shuffle=True,
            generator=generator,
            pin_memory=True,
        )
    val_loader = DataLoader(
            eval_data,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    support_loader = DataLoader(
            support_data,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    return train_loader, val_loader, support_loader


class DOGS(Dataset):

    def __init__(self, path: Path, transform: torch.nn.Module):
        super().__init__()
        self.path = path
        basepath = path.parent.parent / "Images"

        self.df = pd.read_csv(path)
        self._images = []
        self.targets = []
        n_gray = 0
        for _, x in self.df.iterrows():
            with Image.open(basepath / x.file_list, 'r') as img:
                img.load()
            if img.mode == 'L' or img.mode == 'RGBA':
                img = img.convert('RGB')
                n_gray += 1
            self._images.append(img)
            self.targets.append(x.labels)
        LOG.info(f"Number of grayscale images: {n_gray}\nTotal images: {len(self._images)}")

        self.transform = transform

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img = copy.deepcopy(self._images[idx])
        label = self.targets[idx]
        return self.transform(img), label


def get_dogs_dataloader(data_dir, batch_size, num_workers, is_bcos=False):
    data_dir = Path(data_dir)
    generator = torch.Generator().manual_seed(42)
    eval_data = DOGS(data_dir / "valid.csv", EvalTransform(is_bcos=is_bcos))
    support_data = DOGS(data_dir / "train.csv", EvalTransform(is_bcos=is_bcos))
    train_data = DOGS(data_dir / "train.csv", TrainTransform(random_erase_prob=0.5, is_bcos=is_bcos))

    train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            shuffle=True,
            generator=generator,
            pin_memory=True,
        )

    val_loader = DataLoader(
            eval_data,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    support_loader = DataLoader(
            support_data,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    return train_loader, val_loader, support_loader


if __name__ == "__main__":
    import argparse
    import pandas as pd
    import scipy.io as sio
    import numpy as np
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    basepath = Path(args.data_dir)
    train_paths = basepath / "train_list.mat"
    test_paths = basepath / "test_list.mat"
    all_files = basepath / "file_list.mat"
    output = basepath / "processed"
    output.mkdir(exist_ok=True)

    def collapse_values(arr):  # convert MATLAB cell arrays to Python lists
        if isinstance(arr[0], np.ndarray):
            ret = [len(x) == 1 for x in arr]
            assert all(ret)
            return [x[0] for x in arr]
        else:
            return arr.tolist()
        
    train_imgs = sio.loadmat(train_paths)
    train_imgs = {k: collapse_values(v.squeeze()) for k, v in train_imgs.items() if k[0] != "_"}
    train_imgs = pd.DataFrame(train_imgs)
    train_imgs.loc[:, "labels"] = train_imgs.labels - 1
    print(train_imgs.head(), train_imgs.labels.min(), train_imgs.labels.max())

    test_imgs = sio.loadmat(test_paths)
    test_imgs = {k: collapse_values(v.squeeze()) for k, v in test_imgs.items() if k[0] != "_"}
    test_imgs = pd.DataFrame(test_imgs)
    test_imgs.loc[:, "labels"] = test_imgs.labels - 1
    print(test_imgs.head())

    y = train_imgs.labels.to_frame()
    print(len(y.index))

    print(len(train_imgs.index), len(test_imgs.index))
    train_imgs.to_csv(output / "train.csv")
    test_imgs.to_csv(output / "valid.csv")

    # sanity check minimal
    df1 = pd.read_csv(output / "train.csv", index_col=0)
    df2 = pd.read_csv(output / "valid.csv", index_col=0)

    print(len(df1.index), len(df2.index))
    assert len(df1.index.intersection(df2.index)) == 0, f"Number of images in both datasets is {len(df1.index.intersection(df2.index))}"

    print("Successfully processed the dataset!")