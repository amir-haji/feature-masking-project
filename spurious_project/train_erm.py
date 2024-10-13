import torch
import argparse
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import random
import os
import numpy as np
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import weight_init
from data.urbancars import get_urbancars_loaders
from data.domino import get_domino_loaders


def test_cnn(dataset, model, test = False):
    device = torch.device('cuda')
    """
    Conventional testing of a classifier.
    """
    avg_inv_acc = 0
    count = 0

    if test:
        num_envs = 8
    else:
        num_envs = 4
    corrects_envs = [0]*num_envs
    totals_envs = [0] * num_envs
    avg_acc_envs = [0] * num_envs

    model.eval()
    for (batch, (inputs, labels, envs)) in enumerate(tqdm(dataset)):
        count+=1

        inputs = inputs.to(device)
        labels = labels.to(device)
        envs = envs.to(device)
        logits = model(inputs)

        for env_num in range(num_envs):
            logits_env = logits[envs==env_num]
            labels_env = labels[envs==env_num]
            # print (labels_env, torch.argmax(logits_env, dim=1))
            corrects_envs[env_num] += torch.sum(torch.argmax(logits_env, dim=1) == labels_env).item()
            totals_envs[env_num] += len(logits_env)

    all_correct = 0
    all_totals = 0
    for env_num in range(num_envs):
        avg_acc_envs[env_num] = round(corrects_envs[env_num] / totals_envs[env_num], 4)
        print(f"env {env_num}, acc: {avg_acc_envs[env_num]}")
        all_correct += corrects_envs[env_num]
        all_totals += totals_envs[env_num]
    avg_inv_acc = round(all_correct / all_totals, 6)

    print(f"all envs mean acc: {avg_inv_acc}")

    return avg_inv_acc, avg_acc_envs

def train_cnn(dataloader, model, opt, scheduler, step, device):
    criterion =  nn.CrossEntropyLoss()

    ### average loss
    avg_acc = 0
    avg_loss = 0
    count = 0

    model.train()
    for (batch, (inputs, labels, envs)) in enumerate(tqdm(dataloader)):
        count += inputs.shape[0]

        inputs = inputs.to(device)
        # print(labels.shape)
        labels = (F.one_hot(labels, 2).type(torch.FloatTensor)).to(device)

        opt.zero_grad()
        logits = model(inputs)
        total_loss = criterion(logits, labels.float())

        total_loss.backward()
        opt.step()

        avg_loss += total_loss
        avg_acc += torch.sum(torch.argmax(logits, dim=1)==torch.argmax(labels, dim=1))

    # results
    avg_acc = avg_acc/(count)
    avg_loss = avg_loss/(count)

    print("{:s}{:d}: {:s}{:.4f}, {:s}{:.4f}.".format(
        "----> [Train] Total iteration #", step, "acc: ",
        avg_acc, "loss: ", avg_loss),
          flush=True)

    # if scheduler != None:
    #     scheduler.step()

    return step+1


if __name__=="__main__":
    seed = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)

    args = parser.parse_args()

    seed = args.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    root_dir = '/home/f_hosseini/data/dominoesCMF'
    train_loader, retrain_loader, val_loader, test_loader = get_domino_loaders(root = root_dir, batch_size = 128)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet18(pretrained=True)
    d = model.fc.in_features
    model.fc = nn.Linear(d, 2)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 1e-4)
    scheduler = CosineAnnealingLR(
                optimizer,
                T_max=20)


    step = 0
    for i in range (20):
        step = train_cnn(train_loader, model, optimizer, scheduler, step, torch.device('cuda'))

        print ('test')
        test_cnn(test_loader, model, test = True)

    path = f"/home/f_hosseini/dfr-ckpts/dominoes_95/seed{args.seed}_erm.model"
    torch.save(model.state_dict, path)

    print ("erm finished!")
    weight_init(model.fc)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    params = model.fc.parameters()
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=100)


    step = 0
    for i in range(100):
        step = train_cnn(retrain_loader, model, optimizer, scheduler, step, torch.device('cuda'))


        print('test')
        test_cnn(test_loader, model, test = True)

    path = f"/home/f_hosseini/dfr-ckpts/dominoes_95/seed{args.seed}_ll.model"
    torch.save(model.state_dict, path)








