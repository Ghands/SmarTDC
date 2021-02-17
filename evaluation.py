#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

from torch.utils.data import DataLoader
from datasets import TestDataset


class Evaluator(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.NLLLoss().to(self.device)

    def __call__(self, model, dataset):
        # evaluation mode
        model.eval()
        loss, total, correct = 0., 0., 0.
        batch_number = 0

        dataloader = DataLoader(TestDataset(dataset), batch_size=self.args.local_bs, shuffle=False)

        # evaluation
        for _, (images, labels) in enumerate(dataloader):
            batch_number += 1
            images, labels = images.to(self.device), labels.to(self.device)

            # inference
            predicts = model(images)
            batch_loss = self.criterion(predicts, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(predicts, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss / batch_number
