import os
import utils
import data
import torch
import numpy as np
from tqdm import tqdm
import random

def normalize (x):
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    return x

def get_embed(m, x):
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)

    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)

    x = m.avgpool(x)
    x = torch.flatten(x, 1)
    return x

if __name__=='__main__':
    # torch.multiprocessing.set_sharing_strategy('file_system')
    for seed in ([1, 2, 3]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainloader, lastlayerloader, valloader, testloader = data.get_waterbirds_loaders('/home/user01/data/waterbirds/waterbird_complete95_forest2water2/',
                                                                                      batch_size=32)
        sets = {'lastlayer':lastlayerloader, 'val': valloader, 'test': testloader}

        if not os.path.exists(f'/home/user01/data/waterbirds/features_seed{seed}/'):
            os.makedirs(f'/home/user01/data/waterbirds/features_seed{seed}/')
        model = utils.get_pretrained_resnet50(device, pretrained_path=f'/home/user01/dfr-ckpts/waterbirds/erm_seed{seed}/final_checkpoint.pt', mode='dfr')

        for n, loader in sets.items():
            model.eval()

            all_features = []
            all_ys = []
            all_envs = []
            all_ids = []

            with torch.no_grad():
                for batch, (x, y, env) in enumerate(tqdm(loader)):
                    feature = get_embed(model,x.to(device))
                    all_features.append(feature.detach().cpu())
                    all_ys.append(y.detach())
                    all_envs.append(env)

            all_features = torch.concat(all_features, 0)
            all_ys = torch.concat(all_ys, 0)
            all_envs = torch.concat(all_envs, 0)

            print (all_features.shape, all_ys.shape, all_envs.shape)

            torch.save (all_features, f'/home/user01/data/waterbirds/features_seed{seed}/{n}_features.pt')
            torch.save(all_ys,  f'/home/user01/data/waterbirds/features_seed{seed}/{n}_labels.pt')
            torch.save(all_envs, f'/home/user01/data/waterbirds/features_seed{seed}/{n}_envs.pt')
