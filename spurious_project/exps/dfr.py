import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from .experiment import Experiment

class DFR(Experiment):
    def __init__(self):
        super().__init__('DFR')
        
    def create_balanced_dataloader(self, miscls_envs, corrcls_envs, sample_size, **kwargs):
        assert 'batch_size' in kwargs.keys(), 'Missing batch_size in arguments'
        balanced_data = []
        envs = []
        for env_id, miscls_tensors in miscls_envs.items():
            random.shuffle(miscls_tensors)
            selected_tensors = miscls_tensors[:sample_size]
            balanced_data.extend(selected_tensors)
            envs.extend([env_id]*len(selected_tensors))
        for env_id, corrcls_tensors in corrcls_envs.items():
            random.shuffle(corrcls_tensors)
            selected_tensors = corrcls_tensors[:sample_size]
            balanced_data.extend(selected_tensors)
            envs.extend([env_id]*len(selected_tensors))
        features = torch.stack([tensor[0] for tensor in balanced_data])
        labels = torch.stack([tensor[1] for tensor in balanced_data])
        envs = torch.Tensor(envs)
        balanced_dataset = TensorDataset(features, labels, envs)
        balanced_dataloader = DataLoader(balanced_dataset, batch_size=kwargs['batch_size'], shuffle=True)
        return balanced_dataloader