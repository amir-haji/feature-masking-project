import os
import argparse
import utils
import data
import torch
import numpy as np
from tqdm import tqdm
import random
from pytorch_grad_cam.base_cam import BaseCAM
from _test import test_cnn
from exps import LossBasedExp

# /home/f_hosseini/data/multinli/multinli_features/features_noaug_seed1

trainloader, lastlayerloader, valloader, testloader = data.get_feature_loaders(f"/home/f_hosseini/data/haji/multinli/grad_misc_masked/features_seed3/alpha3.2/", 128)

cls_env_dict = {i: [] for i in range(3)}
env_cnt = {i: 0 for i in range(8)}
cls_cnt = {i: 0 for i in range(3)}


for batch, (x, y, env) in enumerate(tqdm(valloader)):
  y = torch.argmax(y, dim = 1)
  env = torch.argmax(env, dim = 1)
  

  
  for i in range(len(y)):
    if env[i].item() not in cls_env_dict[y[i].item()]:
      cls_env_dict[y[i].item()].append(env[i].item())
      
    cls_cnt[y[i].item()] += 1
    env_cnt[env[i].item()] += 1
    
print('*********************')
print(cls_env_dict)
print(cls_cnt)
print(env_cnt)
print('*********************')

