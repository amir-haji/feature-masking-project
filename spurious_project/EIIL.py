from spuco.group_inference import EIIL
import torch
import models
import data
import utils
import argparse
from tqdm import tqdm
import os

def get_dataset_loaders(args):
    '''
        returns trainloader, valloader, testloader with args.batch_size
    '''
    if args.feature_only == True:
        return data.get_feature_loaders(args.dataset_path, args.batch_size)
    elif args.dataset == 'waterbirds':
        return data.get_waterbirds_loaders(args.dataset_path, batch_size=args.batch_size)
    elif args.dataset == 'celeba':
        return data.get_celeba_loaders(args.dataset_path, batch_size=args.batch_size, num_workers=1)
    elif args.dataset == 'dominoe':
        return data.get_dominoes_loaders(batch_size=args.batch_size)
    elif args.dataset == 'cmnist':
        return data.get_cmnist_loaders(args)
    elif args.dataset == 'civilcomments':
        return data.get_civil_comments_loaders(args.pretrained_path, args.dataset_path, args.batch_size)
    elif args.dataset == 'metashift':
        return data.get_metashift_loaders(args.dataset_path, args.batch_size)
    elif args.dataset == 'multinli':
        return data.get_multinli_loaders(args.dataset_path, batch_size=16, num_workers=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EIIL')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--dataset', type=str, default='waterbirds',
                        help='Name of the dataset',
                        choices=['waterbirds', 'celeba', 'multinli', 'dominoe', 'cmnist', 'civilcomments', 'metashift'],
                        required=True)
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to the trained model')
    parser.add_argument('--num_steps', type=int, default=1, help='Number of steps for EIIL')
    parser.add_argument('--feature_only', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.feature_only:
        n = data.dataset_specs.datasets[args.dataset]['num_classes']
        d = data.dataset_specs.datasets[args.dataset]['hidden_layer_size']
        model = utils.get_fc(device, args.pretrained_path, num_features = d, num_classes=n)
    elif args.dataset == 'cmnist':
        model = utils.get_lenet(device, args.pretrained_path)
    elif args.dataset == 'civilcomments':
        model = utils.get_pretrained_bert(args.pretrained_path, 2, device)
    elif args.dataset == 'multinli':
        model = utils.get_pretrained_bert(args.pretrained_path, 3, device)
    elif args.dataset != 'dominoe':
        model = utils.get_pretrained_resnet50(device, args.pretrained_path, mode='dfr')
    else:
        model = utils.get_pretrained_resnet18(device, args.pretrained_path)

    trainloader, lastlayerloader, valloader, testloader = get_dataset_loaders(args)

    inputs = []
    logits = []
    labels = []
    onehot_labels = []
    ground_truth_envs = []

    model.eval()
    for batch, (input, label, env) in enumerate(tqdm(valloader)):
        input = input.to(device)
        label = label.to(device)

        logit = model(input)

        inputs.append(input)
        onehot_labels.append(label)
        labels.append(torch.argmax(label, -1))
        logits.append(logit)
        ground_truth_envs.append(env)

    inputs = torch.cat(inputs, dim=0)
    labels = torch.cat(labels, dim=0)
    onehot_labels = torch.cat(onehot_labels, dim=0)
    logits = torch.cat(logits, dim=0)
    ground_truth_envs = torch.cat(ground_truth_envs, dim=0)

    eiil = EIIL(
        logits=logits,
        class_labels=labels,
        num_steps=args.num_steps,
        lr=args.learning_rate,
        device=device,
        verbose=True
    )

    group_partition = eiil.infer_groups()

    final_partition = {}

    print ('list of groups:')
    for i in range(torch.max(labels).item()+1):
        for j in group_partition.keys():
            final_partition[(i,j)] = []
            print ((i,j))

    for env, indices in group_partition.items():
        print(env)
        for sample in indices:
            final_partition[(labels[sample].item(), env)].append(sample)

    envs = torch.zeros((labels.shape[0], len(final_partition.keys())), dtype=torch.long).to(device)

    for i , (env, indices) in enumerate(final_partition.items()):
        for sample in indices:
            envs[sample][i] = 1


    stats = {}
    for i in range (envs.shape[0]):
        env_label = torch.argmax(envs[i]).item()
        print (env_label)
        if not env_label in stats.keys():
            stats[env_label] = {j: 0 for j in range (ground_truth_envs.shape[-1])}

        stats[env_label][torch.argmax(ground_truth_envs[i]).item()]+=1

    print (stats)

    if not os.path.exists (args.save_path):
        os.makedirs (args.save_path)

    torch.save (inputs, os.path.join(args.save_path, 'val_features.pt'))
    torch.save(onehot_labels, os.path.join(args.save_path, 'val_labels.pt'))
    torch.save(envs, os.path.join(args.save_path, 'val_envs.pt'))