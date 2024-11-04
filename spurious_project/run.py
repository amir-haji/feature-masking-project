from utils import *
from train import *
from _test import *
import torch
import torch.nn as nn
import os
import wandb
import numpy as np


def multi_eval(model, testloaders, log, inferred_groups, args):
    avg_accs = []
    worst_accs = []
    for i, testloader in enumerate(testloaders):
        print (f'Set{i}:')
        acc, worst_acc = test_cnn(testloader, model, log=log, args=args, inferred_groups = inferred_groups)
        avg_accs.append(acc)
        worst_accs.append(worst_acc)

    print (worst_accs)
    return np.mean(avg_accs), np.median(worst_accs)



def run_last_layer_experiment(model, device, balanced_dataloader, testloaders, exp_name,
                              optimizer, l1_lambda, scheduler, dataset='waterbirds',
                              epochs=30, log=False, inspect_loader=None, seed=1, inferred_groups=False, args=None):
    curr_worst = 0
    curr_avg = 0
    if log:
        run = wandb.init(project=exp_name,
                   entity='hesamce',
                   config={
                    "learning_rate": optimizer.state_dict()['param_groups'][0]['initial_lr'],
                    "epochs": epochs,
                    "step_size": scheduler.step_size,
                    "gamma": scheduler.gamma
                },
                  reinit=True
                  )
        
    cnn_optimizer = optimizer
    lr = cnn_optimizer.state_dict()['param_groups'][0]['initial_lr']
    cnn_scheduler = scheduler
    step_size, gamma = cnn_scheduler.step_size, cnn_scheduler.gamma
    global_step = 0
    saved_model = None
    best_model = None
    save_dir = os.path.join(args.output_path,
                            f"{args.experiment}_{args.comments}_{args.dataset}_LR{args.learning_rate}_step{args.step_size}_gamma{args.gamma}_seed{args.seed}_samples{args.sample_size}_l1{args.l1}/")

    for epoch in range(epochs):
        try:
            print("=========================")
            print("epoch:", epoch)
            print("=========================")
            global_step = train_cnn(balanced_dataloader, model, cnn_optimizer, cnn_scheduler, global_step, device, l1_lambda, log)
            print('----> [Val/Test]')
            with torch.no_grad():
                inv_acc, worst_acc = multi_eval(model, testloaders, log, inferred_groups, args)
            if inv_acc > curr_avg:
                curr_avg = inv_acc
                if best_model:
                    os.remove(best_model)
                with torch.no_grad():
                    best_model = os.path.join(save_dir,f"best_avg_epoch{epoch}.model")
                    torch.save(model.state_dict(), best_model)
            if worst_acc == curr_worst and inv_acc > curr_avg:
                if saved_model:
                    os.remove(saved_model)
                with torch.no_grad():
                    saved_model = os.path.join(save_dir,f"best_worst_epoch{epoch}.model")
                    torch.save(model.state_dict(), saved_model)
            if worst_acc > curr_worst:
                curr_worst = worst_acc
                if saved_model:
                    os.remove(saved_model)
                with torch.no_grad():
                    saved_model = os.path.join(save_dir,f"best_worst_epoch{epoch}.model")
                    torch.save(model.state_dict(), saved_model)
            if log:
                wandb.log({"Test Mean Accuracy": inv_acc})
        except KeyboardInterrupt:
            print('Experiment Stopped')
            break
    last_model = os.path.join(save_dir,"last.model")
    torch.save(model.state_dict(), last_model)
    print(f'last model saved at {last_model}')
    return saved_model

def run_loss_inspect_experiment(model, device, spuriousity, balanced_dataloader, testloader, exp_name,
                              optimizer, l1_lambda, scheduler, dataset='waterbirds',
                              epochs=30, log=False, inspect_loader=None):
    losses = [[] for i in range(len(inspect_loader.dataset))]
    loss_fn = nn.CrossEntropyLoss()
    curr_worst = 0
    curr_avg = 0
    cnn_optimizer = optimizer
    lr = cnn_optimizer.state_dict()['param_groups'][0]['initial_lr']
    cnn_scheduler = scheduler
    step_size, gamma = cnn_scheduler.step_size, cnn_scheduler.gamma
    global_step = 0
    for epoch in range(epochs):
        try:
            print("=========================")
            print("epoch:", epoch)
            print("=========================")
            global_step = train_cnn(balanced_dataloader, model, cnn_optimizer, cnn_scheduler, global_step, device, l1_lambda, log)
            print("====== Calculating losses on inspect samples ======")
            with torch.no_grad():
                for (i, (x, y, g)) in enumerate(inspect_loader):
                    y_pred = model(x.cuda())
                    loss = loss_fn(y_pred, y.cuda())
                    losses[i].append(loss.cpu().item())
        except KeyboardInterrupt:
            print('Experiment Stopped')
            break
    last_model = f'last_{exp_name}_{dataset}_sp{spuriousity}_LR{lr}_epoch{global_step}_step{step_size}_gamma{gamma}.model'
    torch.save(model.state_dict(), last_model)
    print(f'last model saved at {last_model}')
    return model, losses
