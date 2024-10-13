import os
import subprocess
from itertools import product
from tqdm.auto import tqdm
import time

sample_size = [500, 750, 1000, 1500]
l1_reg = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
batch_size = [32]
lr = [5e-4]
weight_decays = [1e-4]
scheduler_step_size = [100]
dataset = 'civilcomments2'
dataset_path = '/home/f_hosseini/data/civilcomments_16/features_noaug_seed1'
ll_path = '/home/f_hosseini/data/civilcomments/features_noaug_seed1'
pretrained_path = '/home/f_hosseini/dfr-ckpts/civilcomments/erm_seed1/final_checkpoint.pt'
output_path = '/home/f_hosseini/lfr_logs/civilcomments/16/seed1/'
num_val = [1]
comments = ''
log_path = './civilcomments_16_not_free_seed1_pt2.sh'
with open(log_path, 'w') as f:
    for count, (s_size, reg, bs, l, step_size, wd, nv) in enumerate(product(sample_size, l1_reg, batch_size, lr, scheduler_step_size, weight_decays, num_val)):
            print(' '.join(['python3', 'main.py',\
                            f"--dataset {dataset}",\
                            f"--dataset_path {dataset_path}",\
                            f"--ll_path {ll_path}",\
                            "--experiment loss",\
                            f"--sample_size {s_size}",\
                            f"-b {bs}",\
                            f"-lr {l}",\
                            f"--pretrained_path {pretrained_path}",\
                            f"--gamma 0.5",\
                            f"--weight_decay {wd}",\
                            f"--l1 {reg}",\
                            f"--epochs 50",\
                            f"--optimizer adam",\
                            f"--step_size {step_size}",\
                            f"--seed 1",\
                            f"--output_path {output_path}",\
                            "--validation_path /home/f_hosseini/validation_groups/multinli/seed1",\
                            "--feature_only True"]), file = f, flush=True)