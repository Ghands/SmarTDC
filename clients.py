#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class ClientOutput(object):
    """
    Recommend keys:
        parameter: the state of a client model
        number: the training number of a client model
        epochs: The list of epochs that this client is chosen to aggregation
        accuracy: The list of the accuracy of a client model if it is chosen
        local_loss: The list of the loss of client model on training dataset
        global_loss: The list of the loss of client model on test dataset
        epoch_reward: The list of received reward in an epoch
        local_part: The list of local part reward
        global_part: The list of global part reward
        sum_rewards: all rewards in the task
        malicious: This node is malicious or not in this task.
    """
    def __init__(self, idx):
        self.__dict__.update({
            "idx": idx,
            "number": 0,
            "epochs": [],
            "accuracy": [],
            "local_loss": [],
            "global_loss": [],
            "epoch_reward": [],
            "local_part": [],
            "global_part": [],
            "sum_rewards": 0,
            "malicious": False
        })

    def update(self, **kwargs):
        for key in kwargs:
            if key == "parameter":
                self.__dict__.update(parameter=kwargs[key])
            elif key == "number":
                self.__dict__.update(number=kwargs[key])
            elif key == "sum_rewards":
                self.__dict__.update(sum_rewards=kwargs[key])
            elif key == "malicious":
                self.__dict__.update(malicious=kwargs[key])
            else:
                self.__dict__.get(key).append(kwargs[key])

    def output(self):
        return {
            "idx": self.idx,
            "number": self.number,
            "epochs": self.epochs[-1],
            "accuracy": self.accuracy[-1],
            "local_loss": self.local_loss[-1],
            "global_loss": self.global_loss[-1],
            "epoch_reward": self.epoch_reward[-1],
            "local_part": self.local_part[-1],
            "global_part": self.global_part[-1],
            "sum_rewards": self.sum_rewards
        }



class ChooseClient(object):
    def __init__(self, args, malicious_client_idxs=None):
        self.args = args
        self.malicious_clients = malicious_client_idxs

    def part_pick(self, client_ids, k):
        """ Randomly choose some clients
        
        Malicious nodes must be picked. The reasons are following:
        1. Malicious must participate in the task actively, the server will receive their submitted models.
        2. To ensure their attack can succeed, the performances of their machines are nice with high probability.

        Args: 
            client_ids: a list of client ids
            k: the number of chosen clients

        Return:
            a list of chosen client ids
        """
        if self.args.malicious:
            pick_list = []
            if self.args.malicious:
                pick_list.extend(self.malicious_clients.tolist())

            # delete malicious clients in `client_ids` in case repeated choose
            for to_delete_idx in self.malicious_clients:
                client_ids.pop(to_delete_idx)

            # Randomly choose remain clients
            pick_list.extend(np.random.choice(client_ids, k - len(self.malicious_clients), replace=False).tolist())

            return pick_list
        else:
            return np.random.choice(client_ids, k, replace=False)

    def full_pick(self, client_ids):
        """
        Args:
            client_ids: a list of client ids

        Return:
            a shuffled list of client ids to meet the ability of clients.
        """

        return np.random.shuffle(client_ids)