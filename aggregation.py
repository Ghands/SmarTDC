#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import torch

import numpy as np


class SimpleAggregation(object):
    """This is an abstract aggregation class"""
    def __init__(self, k, f, weight_or_not, no_consensus):
        self.k = k
        self.f = f
        self.weight_flag = weight_or_not
        self.no_consensus = no_consensus

    def choose_clients(self, client_outputs):
        """
        Args:
            client_outputs: A list of outputs produced by clients.

        Return:
            A list of chosen clients
        """
        # Classical federated learning accepts all received models.
        # chosen_clients = np.random.choice(client_outputs, self.k, replace=False)
        return client_outputs

    def without_consensus(self, client_outputs):
        """Randomly choose $f$ models be the blank model.
        """
        # Choose $f$ models to be blank models.
        for single_client in client_outputs[-self.f:]:
            for key in single_client.parameter.keys():
                single_client.parameter[key] = torch.rand_like(single_client.parameter[key])
        return client_outputs

    def weighted_average(self, client_outputs):
        """ 
        Args:
            client_outputs: A list of filtered outpus produced by clients

        Return：
            The parameters of the global model.
        """
        total_clients = len(client_outputs)

        if self.weight_flag:
            all_numbers = [client.number for client in client_outputs]
        else:
            all_numbers = [1] * total_clients

        total_number = sum(all_numbers)
        weights = [client_number / total_number for client_number in all_numbers]

        output = copy.deepcopy(client_outputs[0].parameter)
        for key in output.keys():
            output[key] *= weights[0]
            for i in range(1, total_clients):
                output[key] += weights[i] * client_outputs[i].parameter[key]
            # output[key] = torch.div(output[key], total_clients)
        return output

    def __call__(self, client_outputs):
        chosen_clients = self.choose_clients(client_outputs)
        if self.no_consensus:
            chosen_clients = self.without_consensus(chosen_clients)
        return chosen_clients, self.weighted_average(chosen_clients)


class ThresholdAggregation(SimpleAggregation):
    # def __init__(self, k, weight_or_not):
    #     super(ThresholdAggregation, self).__init__(k, weight_or_not)

    def choose_clients(self, client_outputs):
        """
        Args:
            client_outputs: A list of outputs produced by clients.

        Return:
            A list of chosen clients
        """
        # There exists a threshold in this aggregation mechanism
        # The first k received models will be adopted.
        # Malicious models must be accepted due to the brilliant computing power.
        chosen_clients = client_outputs[:self.k]
        return chosen_clients


class TopAggregation(SimpleAggregation):
    # def __init__(self, k, weight_or_not):
    #     super(TopAggregation, self).__init__(k, weight_or_not)

    def choose_clients(self, client_outputs):
        """
        Args:
            client_outputs: A list of outputs produced by clients.

        Return:
            A list of chosen clients
        """
        # Choose k client models according to the accuracy on test dataset.
        # Should add some random client models in case over-fitting.
        client_outputs.sort(key=lambda temp: temp.accuracy[-1], reverse=True)
        top_num = int(self.k / 2)
        chosen_clients = client_outputs[:top_num]
        chosen_clients.extend(np.random.choice(client_outputs[top_num:], self.k - top_num, replace=False).tolist())
        return chosen_clients