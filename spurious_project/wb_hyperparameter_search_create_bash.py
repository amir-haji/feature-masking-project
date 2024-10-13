import os
import subprocess
from itertools import product
from tqdm.auto import tqdm
import time

sample_size = [20, 25, 30, 35, 40, 45, 50, 55, 60]
l1_reg = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
batch_size = [32]
lr = [5e-4]
weight_decays = [1e-4]
scheduler_step_size = [85]
dataset = 'waterbirds'
dataset_path = '/home/f_hosseini/data/waterbirds/features_noaug_seed1/'
pretrained_path = '/home/f_hosseini/dfr-ckpts/waterbirds/erm_seed1/final_checkpoint.pt'
output_path = '/home/f_hosseini/lfr_logs/waterbirds/eiil/seed1/'
num_val = [1]
comments = ''
log_path = './wb_eiil_seed1.sh'
with open(log_path, 'w') as f:
    for count, (s_size, reg, bs, l, step_size, wd, nv) in enumerate(product(sample_size, l1_reg, batch_size, lr, scheduler_step_size, weight_decays, num_val)):
            print(' '.join(['python3', 'main.py',\
                            f"--dataset {dataset}",\
                            f"--dataset_path {dataset_path}",\
                            "--experiment loss",\
                            f"--sample_size {s_size}",\
                            f"-b {bs}",\
                            f"-lr {l}",\
                            f"--pretrained_path {pretrained_path}",\
                            f"--gamma 0.5",\
                            f"--weight_decay {wd}",\
                            f"--l1 {reg}",\
                            f"--epochs 100",\
                            f"--optimizer adam",\
                            f"--step_size {step_size}",\
                            f"--seed 1",\
                            f"--output_path {output_path}",\
                            "--for_free True",\
                            "--saved_val True",\
                            "--validation_path  /home/f_hosseini/validation_groups/waterbirds/seed1/",\
                            "--feature_only True"]), file = f, flush=True)