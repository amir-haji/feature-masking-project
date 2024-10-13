import utils
import data
import torch
import numpy as np
from tqdm import tqdm

if __name__=='__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, lastlayerloader, valloader, testloader = data.get_multinli_loaders('/home/user01/data/multinli/', batch_size = 16, num_workers = 2)
    sets = {'val': valloader, 'lastlayer': lastlayerloader, 'testloader': testloader}

    model = utils.get_pretrained_bert('/home/user01/dfr-ckpts/multinli/erm_seed1/final_checkpoint.pt', 3, device)
    model.fc = torch.nn.Identity(model.fc.in_features)

    for n, loader in sets.items():
        model.eval()

        all_features = []
        all_ys = []
        all_envs = []
        all_ids = []

        for batch, (x, y, env) in enumerate(tqdm(loader)):
            feature = model(x.to(device))

            all_features.append(feature.detach().cpu())
            all_ys.append(y)
            all_envs.append(env)
            all_ids.append(id)

        all_features = torch.concat(all_features, 0)
        all_ys = torch.concat(all_ys, 0)
        all_envs = torch.concat(all_envs, 0)
        # all_ids = torch.concat(all_ids)

        print (all_features.shape, all_ys.shape, all_envs.shape)

        torch.save (all_features, f'/home/user01/data/multinli_features/{n}_features.pt')
        torch.save(all_ys,  f'/home/user01/data/multinli_features/{n}_labels.pt')
        torch.save(all_envs, f'/home/user01/data/multinli_features/{n}_envs.pt')
        # torch.save(all_ids, f'/home/user01/data/mnli_features/{n}_ids.pt')
