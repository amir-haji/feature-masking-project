import os
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader


def get_feature_dataset(root_dir, split):
    features = torch.load(os.path.join(root_dir, f"{split}_features.pt"))
    labels = torch.load(os.path.join(root_dir, f"{split}_labels.pt"))
    groups = torch.load(os.path.join(root_dir, f"{split}_envs.pt"))

    if split == 'val':
        subsampled_indices = torch.Tensor(np.array([
            0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
            14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
            28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
            42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
            56,  57,  58,  59,  60,  61,  62,  68,  94, 120,  98,  97,  75, 117,
            63, 103, 116, 128, 175, 181, 174, 156, 180, 147, 138, 130, 133, 191,
            196, 198, 240, 247, 194, 216, 243, 207, 230, 281, 261, 291, 280, 278,
            269, 272, 303, 270, 297, 332, 319, 325, 339, 361, 343, 350, 334, 320,
            315, 391, 420, 402, 404, 379, 433, 419, 412, 406, 438, 441, 442, 443,
            444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457,
            458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471,
            472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485,
            486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499,
            500, 501, 502, 503])).type(torch.IntTensor)
        subsampled_features = features[subsampled_indices]
        subsampled_labels = labels[subsampled_indices]
        subsampled_groups = groups[subsampled_indices]

        return TensorDataset(subsampled_features, subsampled_labels, subsampled_groups)

    if split == 'lastlayer':
        subsampled_indices = torch.Tensor(np.array([
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61, 120, 109, 103, 123,  84,  95,  73,  68,
        114,  94, 136, 155, 177, 184, 130, 127, 175, 141, 139, 124, 230, 190,
        186, 212, 191, 232, 206, 209, 197, 193, 292, 255, 263, 306, 307, 249,
        282, 277, 269, 302, 313, 319, 370, 366, 320, 333, 365, 342, 345, 346,
        391, 416, 428, 412, 397, 372, 431, 387, 420, 429, 433, 434, 435, 436,
        437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450,
        451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464,
        465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478,
        479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492,
        493, 494])).type(torch.IntTensor)
        subsampled_features = features[subsampled_indices]
        subsampled_labels = labels[subsampled_indices]
        subsampled_groups = groups[subsampled_indices]

        return TensorDataset(subsampled_features, subsampled_labels, subsampled_groups)


    return TensorDataset(features, labels, groups)


def get_feature_loaders(root_dir, batch_size, num_workers=2):
    # train_loader = DataLoader(get_feature_dataset(root_dir, 'train'), batch_size = batch_size, shuffle = True, num_workers = num_workers)
    lastlayer_loader = DataLoader(get_feature_dataset(root_dir, 'lastlayer'), batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)
    valloader = DataLoader(get_feature_dataset(root_dir, 'val'), batch_size=512, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(get_feature_dataset(root_dir, 'test'), batch_size=512, shuffle=False,
                             num_workers=num_workers)

    return None, lastlayer_loader, valloader, test_loader


def get_feature_loader(root_dir, split, batch_size=128, num_workers=2, shuffle=False):
    loader = DataLoader(get_feature_dataset(root_dir, split), batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers)

    return loader
