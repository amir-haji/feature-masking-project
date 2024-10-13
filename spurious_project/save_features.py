import utils
import data
import torch
import numpy as np
from tqdm import tqdm
import random
import os

if __name__=='__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, lastlayerloader, valloader, testloader = data.get_civil_comments_loaders('bert-base-uncased', '/home/f_hosseini/data/civilcomments', batch_size = 16, num_workers = 2)
    sets = {'val': valloader, 'lastlayer': lastlayerloader, 'test':testloader}

    for seed in ([2, 3]):
        if not os.path.exists(f'/home/f_hosseini/data/civilcomments_16/features_noaug_seed{seed}/'):
            os.makedirs(f'/home/f_hosseini/data/civilcomments_16/features_noaug_seed{seed}/')

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        model = utils.get_pretrained_bert(f'/home/f_hosseini/dfr-ckpts/civilcomments/erm_seed{seed}/final_checkpoint.pt', 2, device)
        model.fc = torch.nn.Identity(model.fc.in_features)

        model.eval()
        for n, loader in sets.items():
            all_features = []
            all_ys = []
            all_envs = []
            all_ids = []

            for batch, (x, y, env, id) in enumerate(tqdm(loader)):
                # print (env)
                feature = model(x.to(device))
                all_features.append(feature.detach().cpu())
                all_ys.append(y)
                all_envs.append(env)
                all_ids.append(id)

            all_features = torch.concat(all_features, 0)
            all_ys = torch.concat(all_ys, 0)
            all_envs = torch.concat(all_envs, 0)
            all_ids = torch.concat(all_ids)

            print (all_features.shape, all_ys.shape, all_envs.shape, all_ids.shape)

            torch.save (all_features, f'/home/f_hosseini/data/civilcomments_16/features_noaug_seed{seed}/{n}_features.pt')
            torch.save(all_ys,  f'/home/f_hosseini/data/civilcomments_16/features_noaug_seed{seed}/{n}_labels.pt')
            torch.save(all_envs, f'/home/f_hosseini/data/civilcomments_16/features_noaug_seed{seed}/{n}_envs.pt')
            torch.save(all_ids, f'/home/f_hosseini/data/civilcomments_16/features_noaug_seed{seed}/{n}_ids.pt')