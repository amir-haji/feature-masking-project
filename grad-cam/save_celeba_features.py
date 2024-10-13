import os
import utils
import data
import torch
import numpy as np
from tqdm import tqdm
import argparse
import random
from pytorch_grad_cam.base_cam import BaseCAM
from _test import test_cnn
from exps import LossBasedExp

def create_mask (envs, heatmaps, alpha):
    val_gs = torch.argmax(envs, -1).cpu().numpy()
    val_grads = heatmaps.cpu().numpy()

    maj_inds = [np.argwhere(val_gs == i).ravel() for i in [0, 3]]
    min_inds = [np.argwhere(val_gs == i).ravel() for i in [1, 2]]


    min_feats_0 = np.argwhere(val_grads[min_inds[0]].mean(0) > alpha * val_grads[maj_inds[0]].mean(0)).ravel()
    min_feats_1 = np.argwhere(val_grads[min_inds[1]].mean(0) > alpha * val_grads[maj_inds[1]].mean(0)).ravel()
    selected_feats = np.union1d(min_feats_0, min_feats_1)

    return selected_feats

def create_heatmap_using_XAI(features, model, heatmap_generator):
    all_heatmaps = []

    features.requires_grad = True

    heatmap = heatmap_generator.grad_fc(features, model, None, use_softmax = False)

    all_heatmaps.append(np.abs(heatmap))

    features.requires_grad = False

    return torch.Tensor(all_heatmaps).squeeze()

def change_dataloader_env_to_JTT(model, dataloader):
     avg_acc, worst_acc, miscls_envs, corrcls_envs = test_cnn(dataloader, model, return_samples=True)
     experiment = LossBasedExp()
     balanced_loader = experiment.create_misc_dataloader(miscls_envs, corrcls_envs,
                                                                           sample_size=64,
                                                                           model=model, batch_size=128,
                                                                           dataloader=dataloader, balanced=False)
     return balanced_loader

def create_mask_balanced_pseudo_labeled_dataloader(dataloader, model, heatmap_generator, alpha):
    balanced_loader = change_dataloader_env_to_JTT(model, dataloader)
    
    model.eval()

    all_features = []
    all_ys = []
    all_envs = []
    all_ids = []
    all_heatmaps = []

    for batch, (x, y, env) in enumerate(tqdm(balanced_loader)):
        heatmaps = create_heatmap_using_XAI(x.to(device), model, heatmap_generator)
        # print (heatmaps.shape)
        feature = x
        all_heatmaps.append(heatmaps)
        all_features.append(feature.detach().cpu())
        all_ys.append(y.detach())
        all_envs.append(env)

    all_features = torch.concat(all_features, 0)
    all_heatmaps = torch.concat(all_heatmaps, 0)
    all_ys = torch.concat(all_ys, 0)
    all_envs = torch.concat(all_envs, 0)
    
    selected_feats = create_mask  (all_envs, all_heatmaps, alpha)
    return selected_feats





if __name__=='__main__':
    # torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description='Feature selection experiment')
    parser.add_argument('--misc', type=bool, default=False, help='misclassification selection')
    args = parser.parse_args()
    
    misc = args.misc
    if misc:
      feat_status = 'grad_misc_masked'
    else:
      feat_status = 'grad_masked'
      
      
    for seed in ([1,2,3]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainloader, lastlayerloader, valloader, testloader = data.get_feature_loaders(f"/home/f_hosseini/data/celeba/features_noaug_seed{seed}", 128)

        sets = {'val': valloader, 'lastlayer':lastlayerloader, 'test': testloader}

        model = utils.get_fc(device, f'/home/f_hosseini/dfr-ckpts/celeba/erm_seed{seed}/final_checkpoint.pt', num_features=2048, num_classes=2)

        heatmap_generator = BaseCAM(
            model=model,
            target_layers=[model],
        )

        for alpha in [1.0, 1.5, 2, 2.6, 3.2]:
            if not os.path.exists(f'/home/f_hosseini/data/haji/celeba/{feat_status}/features_seed{seed}/alpha{alpha}/'):
                os.makedirs(f'/home/f_hosseini/data/haji/celeba/{feat_status}/features_seed{seed}/alpha{alpha}/')

            selected_feats = None
            for n, loader in sets.items():
                model.eval()

                all_features = []
                all_ys = []
                all_envs = []
                all_ids = []
                all_heatmaps = []

                for batch, (x, y, env) in enumerate(tqdm(loader)):
                    heatmaps = create_heatmap_using_XAI(x.to(device), model, heatmap_generator)
                    # print (heatmaps.shape)
                    feature = x
                    all_heatmaps.append(heatmaps)
                    all_features.append(feature.detach().cpu())
                    all_ys.append(y.detach())
                    all_envs.append(env)

                all_features = torch.concat(all_features, 0)
                all_heatmaps = torch.concat(all_heatmaps, 0)
                all_ys = torch.concat(all_ys, 0)
                all_envs = torch.concat(all_envs, 0)

                if n == 'val':
                    if misc:
                      selected_feats = create_mask_balanced_pseudo_labeled_dataloader(loader, model, heatmap_generator, alpha)
                    else:
                      selected_feats = create_mask  (all_envs, all_heatmaps, alpha)
                      
                    print (selected_feats)

                if n in ['val', 'test']:
                    new_features = torch.Tensor(all_features.numpy()[:, selected_feats].copy())
                    print (new_features.shape)

                else:
                    new_features = all_features


                print (new_features.shape, all_ys.shape, all_envs.shape)

                torch.save (new_features, f'/home/f_hosseini/data/haji/celeba/{feat_status}/features_seed{seed}/alpha{alpha}/{n}_features.pt')
                torch.save(all_ys,  f'/home/f_hosseini/data/haji/celeba/{feat_status}/features_seed{seed}/alpha{alpha}/{n}_labels.pt')
                torch.save(all_envs, f'/home/f_hosseini/data/haji/celeba/{feat_status}/features_seed{seed}/alpha{alpha}/{n}_envs.pt')

            np.save(f'/home/f_hosseini/data/haji/celeba/{feat_status}/features_seed{seed}/alpha{alpha}/selected_feats.npy', selected_feats)
