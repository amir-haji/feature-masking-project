import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from .experiment import Experiment
import itertools

class LossBasedExp(Experiment):
    def __init__(self):
        super().__init__('LossBased')
        self.env_dict = {}

    def calculate_cross_entropy(self, envs_samples_dict, model):
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        envs_loss = {}
        with torch.no_grad():
            for env, samples in envs_samples_dict.items():
                env_loss = []
                for sample in samples:
                    input_tensor = torch.Tensor(sample[0]).to(device)
                    label = torch.Tensor(sample[1]).float().to(device)

                    output = model(input_tensor.unsqueeze(0))
                    loss = loss_fn(output[0], label)
                    env_loss.append(loss.item())

                envs_loss[env] = env_loss
                print(f'Completed env{env}')

        return envs_loss


    def select_samples(self, envs_samples_dict, envs_loss_values, sample_size, top=True):
        selected_samples = {}

        for env, loss_values in envs_loss_values.items():
            sorted_indices = sorted(range(len(loss_values)), key=lambda i: loss_values[i], reverse=top)
            top_samples = [envs_samples_dict[env][idx] for idx in sorted_indices[:sample_size]]
            selected_samples[env] = top_samples

        return selected_samples

    def select_samples_random(self, envs_samples_dict, envs_loss_values, sample_size, top=True):
        selected_samples = {}

        for env, loss_values in envs_loss_values.items():
            sorted_indices = list(range(len(loss_values)))
            np.random.shuffle(sorted_indices)
            top_samples = [envs_samples_dict[env][idx] for idx in sorted_indices[:sample_size]]
            selected_samples[env] = top_samples

        return selected_samples

    def merge_dicts(self, dict1, dict2):
        merged_x = []
        merged_y = []
        for env, samples in dict1.items():
            merged_x.extend([sample[0].unsqueeze(0) for sample in samples])
            merged_y.extend([sample[1] for sample in samples])

        for env, samples in dict2.items():
            merged_x.extend([sample[0].unsqueeze(0) for sample in samples])
            merged_y.extend([sample[1] for sample in samples])

        merged_x_tensor = torch.vstack(merged_x)
        merged_y_tensor = torch.vstack(merged_y)
        return merged_x_tensor, merged_y_tensor

    def get_classwise_dict(self, dict1):
        ret = {}
        for i in range (0, len(dict1.keys())-1, 2):
            c = int(i/2)
            ret[c] = dict1[i]+dict1[i+1]

        return ret

    def create_balanced_dataloader_ll(self, miscls_data_dict, corrcls_data_dict, sample_size, model, **kwargs):
        assert 'batch_size' in kwargs.keys(), 'Missing batch_size in arguments'
        miscls_loss_dict = self.calculate_cross_entropy(miscls_data_dict, model)
        corrcls_loss_dict = self.calculate_cross_entropy(corrcls_data_dict, model)

        miscls_data_dict = self.get_classwise_dict(miscls_data_dict)
        corrcls_data_dict = self.get_classwise_dict(corrcls_data_dict)

        miscls_loss_dict = self.get_classwise_dict(miscls_loss_dict)
        corrcls_loss_dict = self.get_classwise_dict(corrcls_loss_dict)

        data_dict = {c: miscls_data_dict[c]+corrcls_data_dict[c] for c in miscls_data_dict.keys()}
        loss_dict = {c: miscls_loss_dict[c]+corrcls_loss_dict[c] for c in miscls_loss_dict.keys()}

        high_loss_selected_samples = self.select_samples(data_dict, loss_dict, sample_size, top=True)
        low_loss_selected_samples = self.select_samples(data_dict, loss_dict, sample_size,
                                                       top=False)



        X, y = self.merge_dicts(high_loss_selected_samples, low_loss_selected_samples)
        envs = torch.zeros((X.shape[0],4))
        dataset = TensorDataset(X, y, envs)
        balanced_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True)
        return balanced_loader
        
        
        
    def create_masked_balanced_dataloader(self, selected_feats, miscls_data_dict, corrcls_data_dict, sample_size, model, balanced = True, **kwargs):
        assert 'batch_size' in kwargs.keys(), 'Missing batch_size in arguments'

        # miscls_data_dict = self.get_classwise_dict(miscls_data_dict)
        # corrcls_data_dict = self.get_classwise_dict(corrcls_data_dict)

        data_dict = {}
        min_length = np.min([len(miscls_data_dict[c]) + len(corrcls_data_dict[c]) for c in miscls_data_dict.keys()])

        for c in miscls_data_dict.keys():
            if balanced:
                misc_indices = list(range(len(miscls_data_dict[c])))
                np.random.shuffle(misc_indices)
                misc = [miscls_data_dict[c][idx] for idx in misc_indices[:min_length]]

                corr = corrcls_data_dict[c]
                sorted_indices = list(range(len(corr)))
                np.random.shuffle(sorted_indices)
                selected_corr = [corr[idx] for idx in sorted_indices[:min_length]]

            else:
                misc = miscls_data_dict[c]
                corr = corrcls_data_dict[c]

                sorted_indices = list(range(len(corr)))
                np.random.shuffle(sorted_indices)
                selected_corr = [corr[idx] for idx in sorted_indices[:len(misc)]]


            new_envs = c * torch.ones((len(misc) + len(selected_corr))).long()
            new_envs = F.one_hot(new_envs, 4)

            data_dict[c] = [misc+selected_corr, new_envs]

        merged_x = []
        merged_y = []
        merged_envs = torch.tensor([])
        for c, data in data_dict.items():
            samples, envs = data
            merged_x.extend([sample[0].unsqueeze(0) for sample in samples])
            merged_y.extend([sample[1] for sample in samples])
            merged_envs = torch.cat((merged_envs, envs))
            

        X = torch.vstack(merged_x)
        y = torch.vstack(merged_y)
        
        X = torch.Tensor(X.detach().cpu().numpy()[:, selected_feats].copy())

        # envs = torch.zeros((X.shape[0], 4))
        dataset = TensorDataset(X, y, merged_envs)
        balanced_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True)
        return balanced_loader
        
    def create_balanced_dataloader(self, miscls_data_dict, corrcls_data_dict, sample_size, model, balanced = True, **kwargs):
        assert 'batch_size' in kwargs.keys(), 'Missing batch_size in arguments'

        # miscls_data_dict = self.get_classwise_dict(miscls_data_dict)
        # corrcls_data_dict = self.get_classwise_dict(corrcls_data_dict)

        data_dict = {}
        min_length = np.min([len(miscls_data_dict[c]) + len(corrcls_data_dict[c]) for c in miscls_data_dict.keys()])

        for c in miscls_data_dict.keys():
            if balanced:
                misc_indices = list(range(len(miscls_data_dict[c])))
                np.random.shuffle(misc_indices)
                misc = [miscls_data_dict[c][idx] for idx in misc_indices[:min_length]]

                corr = corrcls_data_dict[c]
                sorted_indices = list(range(len(corr)))
                np.random.shuffle(sorted_indices)
                selected_corr = [corr[idx] for idx in sorted_indices[:min_length]]

            else:
                misc = miscls_data_dict[c]
                corr = corrcls_data_dict[c]

                sorted_indices = list(range(len(corr)))
                np.random.shuffle(sorted_indices)
                selected_corr = [corr[idx] for idx in sorted_indices[:len(misc)]]


            new_envs = c * torch.ones((len(misc) + len(selected_corr))).long()
            new_envs = F.one_hot(new_envs, len(miscls_data_dict.keys()))

            data_dict[c] = [misc+selected_corr, new_envs]

        merged_x = []
        merged_y = []
        merged_envs = torch.tensor([])
        for c, data in data_dict.items():
            samples, envs = data
            merged_x.extend([sample[0].unsqueeze(0) for sample in samples])
            merged_y.extend([sample[1] for sample in samples])
            merged_envs = torch.cat((merged_envs, envs))
            

        X = torch.vstack(merged_x)
        y = torch.vstack(merged_y)
        
        X = torch.Tensor(X.detach().cpu().numpy().copy())

        # envs = torch.zeros((X.shape[0], 4))
        dataset = TensorDataset(X, y, merged_envs)
        balanced_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True)
        return balanced_loader
               
        
    def create_misc_dataloader(self, miscls_data_dict, corrcls_data_dict, sample_size, model, balanced = True, dst = None, **kwargs):
        assert 'batch_size' in kwargs.keys(), 'Missing batch_size in arguments'

        miscls_data_dict = self.get_classwise_dict(miscls_data_dict)
        corrcls_data_dict = self.get_classwise_dict(corrcls_data_dict)

        data_dict = {}
        min_length = np.min([len(miscls_data_dict[c]) for c in miscls_data_dict.keys()])

        for c in miscls_data_dict.keys():
            if balanced:
                misc_indices = list(range(len(miscls_data_dict[c])))
                np.random.shuffle(misc_indices)
                misc = [miscls_data_dict[c][idx] for idx in misc_indices[:min_length]]

                corr = corrcls_data_dict[c]
                sorted_indices = list(range(len(corr)))
                np.random.shuffle(sorted_indices)
                selected_corr = [corr[idx] for idx in sorted_indices[:min_length]]

            else:
                misc = miscls_data_dict[c]
                corr = corrcls_data_dict[c]

                sorted_indices = list(range(len(corr)))
                np.random.shuffle(sorted_indices)
                selected_corr = [corr[idx] for idx in sorted_indices[:len(misc)]]

            new_envs = None
            if dst == 'multinli':
              if c == 0:
                  new_envs = torch.cat((torch.ones((len(misc))) * 1, torch.ones((len(selected_corr))) * 0)).long()
              elif c == 1:
                  new_envs = torch.cat((torch.ones((len(misc))) * 3, torch.ones((len(selected_corr))) *2)).long()
              else:
                  new_envs = torch.cat((torch.ones((len(misc))) * 5, torch.ones((len(selected_corr))) *4)).long()
              new_envs = F.one_hot(new_envs, 6)
            else:
              if c == 0:
                  new_envs = torch.cat((torch.ones((len(misc))) * 1, torch.ones((len(selected_corr))) * 0)).long()
              else:
                  new_envs = torch.cat((torch.ones((len(misc))) * 2, torch.ones((len(selected_corr))) * 3)).long()
              new_envs = F.one_hot(new_envs, 4)

            data_dict[c] = [misc+selected_corr, new_envs]

        merged_x = []
        merged_y = []
        merged_envs = torch.tensor([])
        for c, data in data_dict.items():
            samples, envs = data
            merged_x.extend([sample[0].unsqueeze(0) for sample in samples])
            merged_y.extend([sample[1] for sample in samples])
            merged_envs = torch.cat((merged_envs, envs))

        X = torch.vstack(merged_x)
        y = torch.vstack(merged_y)


        # envs = torch.zeros((X.shape[0], 4))
        dataset = TensorDataset(X, y, merged_envs)
        balanced_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True)
        return balanced_loader



    def create_misc_masked_dataloader(self, selected_feats ,miscls_data_dict, corrcls_data_dict, sample_size, model, balanced = True, **kwargs):
        assert 'batch_size' in kwargs.keys(), 'Missing batch_size in arguments'

        miscls_data_dict = self.get_classwise_dict(miscls_data_dict)
        corrcls_data_dict = self.get_classwise_dict(corrcls_data_dict)

        data_dict = {}
        min_length = np.min([len(miscls_data_dict[c]) for c in miscls_data_dict.keys()])

        for c in miscls_data_dict.keys():
            if balanced:
                misc_indices = list(range(len(miscls_data_dict[c])))
                np.random.shuffle(misc_indices)
                misc = [miscls_data_dict[c][idx] for idx in misc_indices[:min_length]]

                corr = corrcls_data_dict[c]
                sorted_indices = list(range(len(corr)))
                np.random.shuffle(sorted_indices)
                selected_corr = [corr[idx] for idx in sorted_indices[:min_length]]

            else:
                misc = miscls_data_dict[c]
                corr = corrcls_data_dict[c]

                sorted_indices = list(range(len(corr)))
                np.random.shuffle(sorted_indices)
                selected_corr = [corr[idx] for idx in sorted_indices[:len(misc)]]

            new_envs = None
            
            if c == 0:
                new_envs = torch.cat((torch.ones((len(misc))) * 1, torch.ones((len(selected_corr))) * 0)).long()
            else:
                new_envs = torch.cat((torch.ones((len(misc))) * 2, torch.ones((len(selected_corr))) * 3)).long()
            
            
            new_envs = F.one_hot(new_envs, 4)

            data_dict[c] = [misc+selected_corr, new_envs]

        merged_x = []
        merged_y = []
        merged_envs = torch.tensor([])
        for c, data in data_dict.items():
            samples, envs = data
            merged_x.extend([sample[0].unsqueeze(0) for sample in samples])
            merged_y.extend([sample[1] for sample in samples])
            merged_envs = torch.cat((merged_envs, envs))

        X = torch.vstack(merged_x)
        y = torch.vstack(merged_y)

        X = torch.Tensor(X.detach().cpu().numpy()[:, selected_feats].copy())


        # envs = torch.zeros((X.shape[0], 4))
        dataset = TensorDataset(X, y, merged_envs)
        balanced_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True)
        return balanced_loader

        return balanced_loader

    def create_class_balanced_dataloader(self, miscls_data_dict, corrcls_data_dict, sample_size, model, **kwargs):
        assert 'batch_size' in kwargs.keys(), 'Missing batch_size in arguments'

        miscls_data_dict = self.get_classwise_dict(miscls_data_dict)
        corrcls_data_dict = self.get_classwise_dict(corrcls_data_dict)

        data_dict = {c: miscls_data_dict[c]+corrcls_data_dict[c] for c in miscls_data_dict.keys()}
        selected_dict = {}
        sample_size = np.min([len(x) for x in data_dict.values()])

        for c, samples in data_dict.items():
            print (c)
            sorted_indices = list(range(len(samples)))
            np.random.shuffle(sorted_indices)
            selected_dict[c] = [data_dict[c][idx] for idx in sorted_indices[:sample_size]]

        merged_x = []
        merged_y = []

        for c, samples in selected_dict.items():
            merged_x.extend([sample[0].unsqueeze(0) for sample in samples])
            merged_y.extend([sample[1] for sample in samples])

        X = torch.vstack(merged_x)
        y = torch.vstack(merged_y)

        envs = torch.zeros((X.shape[0], 4))
        dataset = TensorDataset(X, y, envs)
        balanced_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True)
        return balanced_loader


    def create_balanced_dataloader_val(self, miscls_data_dict, corrcls_data_dict, sample_size, model, **kwargs):
        assert 'batch_size' in kwargs.keys(), 'Missing batch_size in arguments'
        miscls_loss_dict = self.calculate_cross_entropy(miscls_data_dict, model)
        corrcls_loss_dict = self.calculate_cross_entropy(corrcls_data_dict, model)

        miscls_data_dict = self.get_classwise_dict(miscls_data_dict)
        corrcls_data_dict = self.get_classwise_dict(corrcls_data_dict)

        miscls_loss_dict = self.get_classwise_dict(miscls_loss_dict)
        corrcls_loss_dict = self.get_classwise_dict(corrcls_loss_dict)

        miscls_selected_samples = self.select_samples(miscls_data_dict, miscls_loss_dict, sample_size, top=True)
        corrcls_selected_samples = self.select_samples(corrcls_data_dict, corrcls_loss_dict, sample_size, top=False)

        X, y = self.merge_dicts(miscls_selected_samples, corrcls_selected_samples)

        self.set_env_dict(len(miscls_data_dict.keys()))

        if kwargs.get('for_free', False):
            mis_envs = []
            corr_envs = []

            for i in (miscls_data_dict.keys()):
                l = len(miscls_data_dict[i])
                mis_envs.append(torch.vstack([self.env_dict[(i,0)] for _ in range(l)]) if l else torch.empty(0))

                l = len(corrcls_data_dict[i])
                corr_envs.append(torch.vstack([self.env_dict[(i,1)] for _ in range(l)]) if l else torch.empty(0))

            all_envs = mis_envs + corr_envs

            all_envs = torch.cat(all_envs, 0)

        dataset = TensorDataset(X, y, all_envs)
        balanced_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True)
        return balanced_loader

    def create_balanced_random_dataloader(self, samples: dict, **kwargs):
        """
        randomly group samples in each class into num_group_per_cls groups.
        """
        assert 'batch_size' in kwargs.keys(), 'Missing batch_size in arguments'
        self.set_env_dict(len(samples.keys()))
        all_envs = []
        for c in samples.keys():
            random_groups = np.random.randint(2, size=len(samples[0]))
            g = torch.vstack([self.env_dict[(0, random_groups[i])] for i in range(len(samples[c]))])
            all_envs.append(g)

        envs = torch.cat(all_envs)

        X, y = self.merge_dicts(samples, {})
        dataset = TensorDataset(X, y, envs)
        balanced_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True)
        return balanced_loader


    # selects low-loss samples of correctly classified samples
    def create_biased_dataloader(self, corrcls_envs, sample_size, model, **kwargs):
        assert 'batch_size' in kwargs.keys(), 'Missing batch_size in arguments'
        corrcls_loss_values = self.calculate_cross_entropy(corrcls_envs, model)
        corrcls_label_loss_dict = {0: corrcls_loss_values[0]+corrcls_loss_values[1],
                                1: corrcls_loss_values[2]+corrcls_loss_values[3]}
        corrcls_data_dict = {0: corrcls_envs[0]+corrcls_envs[1],
                            1: corrcls_envs[2]+corrcls_envs[3]}
        corrcls_selected_samples = self.select_samples(corrcls_data_dict, corrcls_label_loss_dict, sample_size, top=False)
        X, y = self.merge_dicts({}, corrcls_selected_samples)
        dummy_envs = torch.zeros((X.shape[0],4))
        dataset = TensorDataset(X, y, dummy_envs)
        biased_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True)
        return biased_loader

    def set_env_dict(self, num_classes):
        index = 0
        for i in range(num_classes):
            for j in range (2):
                self.env_dict[(i,j)] = torch.zeros(num_classes*2)
                self.env_dict[(i,j)][index] = 1
                index += 1

