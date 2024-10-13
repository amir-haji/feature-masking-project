import torch
import numpy as np
import pickle
import os
from torch.utils.data import Dataset, DataLoader


class Domino(Dataset):
    def __init__(self, split, root):
        self.split = split
        if split == 'train':
            with open(os.path.join(root, 'new_train.pkl'), 'rb') as f:
                ind = pickle.load(f)

        elif split == 'val':
            with open(os.path.join(root,'lastlayer_indices_07095.pkl'), 'rb') as f:
                ll = pickle.load(f)
            with open(os.path.join(root,'val_indices_07095.pkl'), 'rb') as f:
                val = pickle.load(f)
            # with open(os.path.join(root, 'val_indices_07095.pkl'), 'rb') as f:
            #     ind = pickle.load(f)
            ind = ll+val

        elif split == 'retrain':
            with open(os.path.join(root, 'new_retrain.pkl'), 'rb') as f:
                ind = pickle.load(f)

        elif split == 'test':
            with open(os.path.join(root, 'test_indices_07095.pkl'), 'rb') as f:
                ind = pickle.load(f)

        X = torch.load(os.path.join(root, 'cifar_cmnist_fmnist_dominoe_X_07095.pt'))
        Y = torch.load(os.path.join(root, 'cifar_cmnist_fmnist_dominoe_Y_07095.pt'))
        G = torch.load(os.path.join(root, 'cifar_cmnist_fmnist_dominoe_G_07095.pt'))

        self.X, self.Y, self.G = X[ind], Y[ind], G[ind]

        self.Y = self.Y.long()
        self.G = self.G.long()

        if self.split in ['train', 'val', 'retrain']:
            self.n_groups = 4
        else:
            self.n_groups = 8

        self.n_classes = 2

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        assert self.split in ['train', 'val', 'test', 'retrain']
        if self.split in ['train', 'val', 'retrain']:
            return self.X[idx], self.Y[idx], self.G[idx] // 2

        return self.X[idx], self.Y[idx], self.G[idx]

    def group_counts(self):
        counts = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(8):
            counts[i] = torch.sum(self.G == i)
        if self.split in ['train', 'val', 'retrain']:
            new_counts = [counts[i] + counts[i + 1] for i in range(0, 8, 2)]
        else:
            new_counts = counts

        return torch.tensor(np.array(new_counts))

    def group_str(self, idx):
        return str(idx)


def get_domino_datasets(root):
    train_set = Domino(root=root, split="train")
    val_set = Domino(root=root, split="val")
    test_set = Domino(root=root, split="test")
    retrain_set = Domino(root=root, split="retrain")

    return train_set, retrain_set, val_set, test_set


def get_domino_loaders(root, batch_size):
    train_set, retrain_set, val_set, test_set = get_domino_datasets(root)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    retrainloader = torch.utils.data.DataLoader(retrain_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, retrainloader, valloader, testloader

