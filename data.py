import torch
from torch.utils.data import random_split, RandomSampler, DataLoader


class DataProcessor:

    def __init__(self, dataset_path):
        self.dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if isinstance(dataset_path, str):
            self.dataset = torch.load(dataset_path)
        elif isinstance(dataset_path, dict):
            if dataset_path.get('train_dataset', None) is not None:
                self.dataset = torch.load(dataset_path['train_dataset'])
            if dataset_path.get('val_dataset', None) is not None:
                self.val_dataset = torch.load(dataset_path['val_dataset'])
            if dataset_path.get('test_dataset', None) is not None:
                self.test_dataset = torch.load(dataset_path['test_dataset'])

    def prepare_dataset(self, train_set_split_prop: int = 0.87, val_set_split_prop: int = 0.13,
                        test_set_split_prop: int = 0.0, batch_size: int = 32):
        train_dataloader, val_dataloader, test_dataloader = None, None, None
        train_subset, val_subset, test_subset = None, None, None

        nb_train_samples = int(train_set_split_prop * len(self.dataset)) if self.dataset is not None else 0
        nb_val_samples = int(val_set_split_prop * len(self.dataset)) if self.dataset and self.val_dataset is None else 0
        nb_test_samples = len(self.dataset) - nb_train_samples - nb_val_samples if self.dataset and self.test_dataset is None else 0

        if self.dataset is not None:
            train_subset, val_subset, test_subset = random_split(self.dataset,
                                                                 [nb_train_samples, nb_val_samples, nb_test_samples])

            train_sampler = RandomSampler(train_subset)
            train_dataloader = DataLoader(train_subset, sampler=train_sampler, batch_size=batch_size)

        if nb_val_samples and val_subset:
            val_sampler = RandomSampler(val_subset)
            val_dataloader = DataLoader(val_subset, sampler=val_sampler, batch_size=batch_size)
        elif self.val_dataset is not None:
            val_dataloader = DataLoader(self.val_dataset, shuffle=True, batch_size=batch_size)

        if nb_test_samples and test_subset:
            test_sampler = RandomSampler(test_subset)
            test_dataloader = DataLoader(test_subset, sampler=test_sampler, batch_size=batch_size)
        elif self.test_dataset is not None:
            test_dataloader = DataLoader(self.test_dataset, shuffle=True, batch_size=batch_size)

        return train_dataloader, val_dataloader, test_dataloader
