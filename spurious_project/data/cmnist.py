from types import SimpleNamespace
from .cnc import initialize_data

args = SimpleNamespace(
    dataset = 'colored_mnist',
    train_encoder = True,
    data_cmap = 'hsv',
    test_shift = 'random',
    p_correlation = 0.995,
    bs_trn_s = 32,
    bs_trn = 32,
    bs_val = 32,
    target_sample_ratio = 1,
    num_workers = 0,
    replicate = 42,
    seed = 42,
    train_classes = [[0,1],[2,3],[4,5],[6,7],[8,9]],
    train_class_ratios=[[1.0],[1.0],[1.0],[1.0],[1.0]],
    val_split=0.2,
    p_corr_by_class=None, #[[0.9],[0.9],[0.9],[0.9],[0.9]],
    flipped=False,
    test_cmap='',
    display_image=True
)

def get_cmnist_loaders(global_args):
    args.seed = global_args.seed
    load_dataloaders = initialize_data(args)
    train_loader, lastlayer_loader, val_loader, test_loader = load_dataloaders(args)
    return train_loader, lastlayer_loader, val_loader, test_loader