#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset.data[self.idxs[item]], self.dataset.targets[self.idxs[item]]
        return image, label


class TestDataset(Dataset):
    """A type of dataset for test dataset
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset.data[item], self.dataset.targets[item]
        return image, label


class SavedData(object):
    def __init__(self, train_dataset, test_dataset, train_clients):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_clients = train_clients


class CreateMNIST(object):
    """Create MNIST dataset for each user"""
    def __init__(self, num_users, data_dir='../data/mnist'):
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081))
        ])

        self.train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        self.test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
        self.num_users = num_users

        train_shape = self.train_dataset.data.shape
        self.train_dataset.data = self.train_dataset.data.reshape([train_shape[0], 1, train_shape[1], train_shape[2]])
        test_shape = self.test_dataset.data.shape
        self.test_dataset.data = self.test_dataset.data.reshape([test_shape[0], 1, test_shape[1], test_shape[2]])

    def create_iid(self, save_filename=None):
        num_items = int(len(self.train_dataset) / self.num_users)
        dict_users  = {} 
        all_idxs = [i for i in range(len(self.train_dataset))]

        for i in range(self.num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        
        # torch.save(SavedData(self.train_dataset, self.test_dataset, dict_users), save_filename)
        return dict_users

    def create_noniid(self, save_filename=None):
        num_shards, num_imgs = 200, 300
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([]) for i in range(self.num_users)}
        idxs = np.arange(num_shards * num_imgs)
        labels = self.train_dataset.targets.numpy()

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign 2 shards/client
        for i in range(self.num_users):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand+1) * num_imgs]), axis=0)
        
        # torch.save(SavedData(self.train_dataset, self.test_dataset, dict_users), save_filename)
        return dict_users

    def crate_noniid_unequal(self, save_filename=None):
        num_shards, num_imgs = 1200, 50
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([]) for i in range(self.num_users)}
        idxs = np.arange(num_shards*num_imgs)
        labels = self.train_dataset.targets.numpy()

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # Minimum and maximum shards assigned per client:
        min_shard = 1
        max_shard = 30

        # Divide the shards into random chunks for every client
        # s.t the sum of these chunks = num_shards
        random_shard_size = np.random.randint(min_shard, max_shard+1,
                                            size=self.num_users)
        random_shard_size = np.around(random_shard_size /
                                    sum(random_shard_size) * num_shards)
        random_shard_size = random_shard_size.astype(int)

        # Assign the shards randomly to each client
        if sum(random_shard_size) > num_shards:

            for i in range(self.num_users):
                # First assign each client 1 shard to ensure every client has
                # atleast one shard of data
                rand_set = set(np.random.choice(idx_shard, 1, replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)

            random_shard_size = random_shard_size-1

            # Next, randomly assign the remaining shards
            for i in range(self.num_users):
                if len(idx_shard) == 0:
                    continue
                shard_size = random_shard_size[i]
                if shard_size > len(idx_shard):
                    shard_size = len(idx_shard)
                rand_set = set(np.random.choice(idx_shard, shard_size,
                                                replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)
        else:

            for i in range(self.num_users):
                shard_size = random_shard_size[i]
                rand_set = set(np.random.choice(idx_shard, shard_size,
                                                replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)

            if len(idx_shard) > 0:
                # Add the leftover shards to the client with minimum images:
                shard_size = len(idx_shard)
                # Add the remaining shard to the client with lowest data
                k = min(dict_users, key=lambda x: len(dict_users.get(x)))
                rand_set = set(np.random.choice(idx_shard, shard_size,
                                                replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[k] = np.concatenate(
                        (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)

        # torch.save(SavedData(self.train_dataset, self.test_dataset, dict_users), save_filename)
        return dict_users


class CreateCIFAR10(object):
    def __init__(self, num_users, data_dir='../data/cifar'):
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        self.test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
        self.num_users = num_users

        train_shape = self.train_dataset.data.shape
        self.train_dataset.data = self.train_dataset.data.reshape([train_shape[0], train_shape[3], train_shape[1], train_shape[2]])
        test_shape = self.test_dataset.data.shape
        self.test_dataset.data = self.test_dataset.data.reshape([test_shape[0], test_shape[3], test_shape[1], test_shape[2]])

    def create_iid(self, save_filename=None):
        num_items = int(len(self.train_dataset)/ self.num_users)
        dict_users, all_idxs = {}, [i for i in range(len(self.train_dataset))]
        for i in range(self.num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])

        # torch.save(SavedData(self.train_dataset, self.test_dataset, dict_users), save_filename)
        return dict_users

    def create_noniid(self, save_filename=None):
        num_shards, num_imgs = 200, 250
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([]) for i in range(self.num_users)}
        idxs = np.arange(num_shards*num_imgs)
        # labels = dataset.targets.numpy()
        labels = np.array(self.train_dataset.targets)

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign
        for i in range(self.num_users):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        # torch.save(SavedData(self.train_dataset, self.test_dataset, dict_users), save_filename)
        return dict_users