import os
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

def get_feature_dataset(root_dir, split):
    features = torch.load(os.path.join(root_dir, f"{split}_features.pt"))
    labels = torch.load(os.path.join(root_dir, f"{split}_labels.pt"))
    groups = torch.load(os.path.join(root_dir, f"{split}_envs.pt"))

    return TensorDataset(features, labels, groups)

def get_feature_loaders(root_dir, batch_size, ll_dir = None, num_workers = 2):
    # train_loader = DataLoader(get_feature_dataset(root_dir, 'train'), batch_size = batch_size, shuffle = True, num_workers = num_workers)
    if ll_dir is None:
        lastlayer_loader = DataLoader(get_feature_dataset(root_dir, 'lastlayer'), batch_size = batch_size, shuffle = True, num_workers = num_workers)
    else:
        lastlayer_loader = DataLoader(get_feature_dataset(ll_dir, 'lastlayer'), batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers)
    valloader = DataLoader(get_feature_dataset(root_dir, 'val'), batch_size=512, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(get_feature_dataset(root_dir, 'test'), batch_size = 512, shuffle = False, num_workers = num_workers)

    return None, lastlayer_loader, valloader, test_loader

def get_feature_loader (root_dir, split, batch_size=512, num_workers = 2, shuffle = False):
    loader = DataLoader(get_feature_dataset(root_dir, split), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader