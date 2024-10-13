# from: https://raw.githubusercontent.com/facebookresearch/Whac-A-Mole/main/dataset/urbancars.py
"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import glob
import torch
import random
import numpy as np
import pandas as pd


from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UrbanCars(Dataset):
    base_folder = "urbancars_images"

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
    ):
        self.split_dict = {'train': 0, 'val': 1, 'test': 2, 'retrain': 3}
        super().__init__()
        self.transform = transform
        self.n_classes = 2
        self.split = split

        if split in ["train", "val", "retrain"]:
            self.n_groups = 4
        else:
            self.n_groups = 8

        metadata = pd.read_csv ('/home/f_hosseini/data/urbancars_images/metadata2.csv')
        self.metadata_df = metadata[metadata['split'] == self.split_dict[split]]
        print (len(self.metadata_df))
        self.group_array = np.array (self.metadata_df['group'].values)
        self.y = np.array (self.metadata_df['label'].values)
        self.img_fpath_list = self.metadata_df['filenames'].values

    def group_counts(self):
        counts = [0,0,0,0,0,0,0,0]
        for i in range(8):
            counts[i] = np.sum(self.group_array==i)
        if self.split in ['train', 'val']:
            new_counts = counts[:4]
        else:
            new_counts = counts

        return torch.tensor(np.array(new_counts))

    def group_str(self,idx):
        return str(idx)

    def get_group_array(self):
        return self.group_array

    def get_label_array(self):
        return self.y


    def __len__(self):
        return len(self.img_fpath_list)

    def __getitem__(self, index):
        img_fpath = self.img_fpath_list[index]
        img_fpath = img_fpath.replace('/home/ghaznavi/data/urbancars_images','/home/f_hosseini/data/urbancars_images')
        y = self.y[index]

        img = Image.open(img_fpath)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

            return img, y, self.group_array[index]

        return img, y, self.group_array[index]

    def get_labels(self):
        return self.obj_bg_co_occur_obj_label_list

    def get_sampling_weights(self):
        group_counts = (
            (torch.arange(self.num_group).unsqueeze(1) == self.group_array)
            .sum(1)
            .float()
        )
        group_weights = len(self) / group_counts
        weights = group_weights[self.group_array]
        return weights

def get_transforms(arch, is_training):
    if arch.startswith("resnet"):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if is_training:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            )
    else:
        raise NotImplementedError

    return transform

def _get_train_loader(batch_size, train_set):

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=None,
        )
        return train_loader

def _get_train_transform():
    train_transform = get_transforms("resnet50", is_training=True)
    return train_transform

def get_urbancars_loaders(batch_size, group_label, root):
    train_transform = _get_train_transform()
    test_transform = get_transforms("resnet50", is_training=False)
    train_set = UrbanCars(
        root,
        "train",
        transform=train_transform,
    )

    retrain_set = UrbanCars(
        root,
        "retrain",
        transform=train_transform,
    )

    val_set = UrbanCars(
        root,
        "val",
        transform=test_transform,
    )

    test_set = UrbanCars(
        root,
        "test",
        transform=test_transform,
    )

    num_class = 2

    train_loader = _get_train_loader(batch_size, train_set)
    retrain_loader = torch.utils.data.DataLoader(
        retrain_set,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, retrain_loader, val_loader, test_loader


def get_urbancars_datasets(batch_size, group_label, root):
    train_transform = _get_train_transform()
    test_transform = get_transforms("resnet50", is_training=False)
    train_set = UrbanCars(
        root,
        "train",
        transform=train_transform,
    )

    val_set = UrbanCars(
        root,
        "val",
        transform=test_transform,
    )

    test_set = UrbanCars(
        root,
        "test",
        transform=test_transform,
    )

    return train_set, val_set, test_set
