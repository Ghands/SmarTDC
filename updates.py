#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import DatasetSplit
from stopping import EarlyStopping

class LocalUpdate(object):
    """ Local update method"""
    def __init__(self, args, dataset, idxs, logger):
        """
        Args:
            args: a collection of all project parameters
            dataset: the full training dataset
            index: indicate the data belong to this client
            logger: an object to record logs
        """
        self.args = args
        self.logger = logger
        self.dataset = dataset
        self.idxs = list(idxs)
        self.train_loader, self.valid_loader = self.split_train_dataset()
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.NLLLoss().to(self.device)

    def split_train_dataset(self):
        # split the idxs into training and evaluation
        split_pos = int(self.args.local_train_rate * len(self.idxs))
        idxs_train = self.idxs[:split_pos]
        idxs_val = self.idxs[split_pos:]

        # get dataloaders
        train_loader = DataLoader(DatasetSplit(self.dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        valid_loader = DataLoader(DatasetSplit(self.dataset, idxs_val), batch_size=int(len(idxs_val) / 10), shuffle=False)

        return train_loader, valid_loader

    def get_optimizer(self, model):
        """
        Args:
            model: a pytorch model.

        Return:
            A optimizer of model parameters
        """
        if self.args.optimizer == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        else:
            raise ValueError('Unsupported optimizer')

    def __call__(self, model):
        """
        Args:
            model: a pytorch model
            dataloader: a training dataset loader of a client

        Return:
            All parameters of trained model.
        """
        # prepare optimizer and early stopping
        model.train()
        optimizer = self.get_optimizer(model)
        early_stopping = EarlyStopping(patience=self.args.stopping_rounds)

        # training process
        epoch_loss = []
        for _ in range(self.args.local_ep):
            # training part
            model.train()
            batch_loss = []
            for _, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # print(sum(batch_loss) / len(batch_loss))

            # validation and early stopping part
            model.eval()
            valid_loss = []
            total, correct =0., 0.
            for _, (images, labels) in enumerate(self.valid_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                valid_loss.append(loss.item())

                # _, pred_labels = torch.max(log_probs, 1)
                # pred_labels = pred_labels.view(-1)
                # correct += torch.sum(torch.eq(pred_labels, labels)).item()
                # total += len(labels)

            # print(correct / total)
            early_stopping(sum(valid_loss) / len(valid_loss), model)
            if early_stopping.early_stop:
                break
        # print("------------------")

        return model, sum(epoch_loss) / len(epoch_loss)