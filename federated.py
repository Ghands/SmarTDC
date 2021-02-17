#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tqdm
import copy
import logging
import torch
import pickle
from time import strftime, localtime
from datasets import CreateMNIST, CreateCIFAR10
from models import CNNMnist, CNNFashion_Mnist, CNNCifar, MLP
from clients import ChooseClient, ClientOutput
from updates import LocalUpdate
from aggregation import SimpleAggregation, ThresholdAggregation, TopAggregation
from evaluation import Evaluator
from incentive import epoch_incentive
from malicious import MaliciousChange
from stopping import EarlyStopping

def federated_main(args):
    # prepare logger
    output_dir = "../logs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    session_time = strftime("%y%m%d-%H%M%S", localtime())
    output_file = os.path.join(output_dir, "{}-{}.log".format(session_time, args.dataset))
    log_file = logging.FileHandler(output_file)
    log_file.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(log_file)

    # prepare client data output directory, related output is at bottom
    if not os.path.exists("../clients_data"):
        os.makedirs("../clients_data")

    # prepare devices
    if args.gpu:
        torch.cuda.set_device("cuda:{}".format(args.gpu))
    device = "cuda" if args.gpu else "cpu"

    # prepare dataset
    if args.dataset == "cifar":
        dataset = CreateCIFAR10(args.num_users)
    elif args.dataset == "mnist":
        dataset = CreateMNIST(args.num_users)
    elif args.dataset == "fmnist":
        dataset = CreateMNIST(args.num_users, "../data/fmnist")
    else:
        raise ValueError("The dataset you input is not supported yet")

    if args.iid:
        user_groups = dataset.create_iid()
    else:
        if args.unequal:
            if args.dataset == "cifar":
                raise NotImplementedError("Cifar 10 unequal dataset is not implemented")
            else:
                user_groups = dataset.crate_noniid_unequal()
        else:
            user_groups = dataset.create_noniid()

    # initialize the model
    if args.model == "cnn":
        if args.dataset == "mnist":
            global_model = CNNMnist(args=args)
        elif args.dataset == "fmnist":
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == "cifar":
            global_model = CNNCifar(args=args)
        else:
            raise ValueError("The dataset you input is not supported yet")
    elif args.model == "mlp":
        img_size = dataset.train_dataset.data[0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        raise ValueError("The model you input is not supported yet")

    global_model.to(device)
    global_model.train()

    # prepare the parameters of global model
    global_parameters = global_model.state_dict()

    # Prepare the client container
    local_outputs = [ClientOutput(client_idx) for client_idx in range(args.num_users)]

    # malicious behavior, some nodes may perform model posion attack by changing the label of data
    if args.malicious:
        malicious_changer = MaliciousChange(args, dataset.train_dataset)
        # the length of source_list and dest_list can influence the final accuracy
        malicious_changer.data_poision(user_groups, [1, 5, 3], [3, 1, 5])
        malicious_client_idxs = malicious_changer.malicious_clients
        for idx in malicious_client_idxs:
            local_outputs[idx].update(malicious=True)

    # Prepare global early stopping
    if args.global_early_stop:
        global_early_stopping = EarlyStopping(patience=args.global_early_rounds)

    # Prepare the aggregation method
    if args.aggre == "simple":
        aggre_method = SimpleAggregation(
            int(args.rate * args.num_users), 
            int((args.num_users - 1) / 3), 
            True, 
            args.no_consensus)
    elif args.aggre == "time":
        aggre_method = ThresholdAggregation(
            int(args.rate * args.num_users), 
            int((args.num_users - 1) / 3), 
            True, 
            args.no_consensus)
    elif args.aggre == "top":
        aggre_method = TopAggregation(
            int(args.rate * args.num_users), 
            int((args.num_users - 1) / 3), 
            False, 
            args.no_consensus)
    else:
        raise ValueError("The aggregation method you indicate has not been implemented")

    # prepare other things
    old_global_metric = 0.
    server_total_profit = 0.
    epoch_outputs = []
    if args.malicious:
        client_chooser = ChooseClient(args, malicious_client_idxs)
    else:
        client_chooser = ChooseClient(args)

    # Training
    for epoch in tqdm.tqdm(range(args.epochs)):
        # Prepare some things
        logger.info("Epoch {} Started".format(epoch))

        # Choose clients participates in this epoch
        if args.aggre == "simple":
            # The number of received models will influence the performance
            # In order to perform a fair comparison, FedAvg has no random screening
            k = max(int(args.rate * args.num_users), 1)
        else:
            k = max(int(args.frac * args.num_users), 1)
        chosen_clients = client_chooser.part_pick(list(range(args.num_users)), k)

        # Client part
        for client_idx in chosen_clients:
            # Client model training
            client_updater = LocalUpdate(args, dataset.train_dataset, user_groups[client_idx], logger)
            client_model, client_loss = client_updater(copy.deepcopy(global_model))

            # Client model evaluation
            evaluator = Evaluator(args, logger)
            client_global_metric, client_global_loss = evaluator(client_model, dataset.test_dataset)

            # Update ClientOutput
            local_outputs[client_idx].update(
                parameter=copy.deepcopy(client_model.state_dict()), 
                number=len(user_groups[client_idx]), 
                epochs=epoch,
                accuracy=client_global_metric, 
                local_loss=client_loss, 
                global_loss=client_global_loss)
            epoch_outputs.append(local_outputs[client_idx])
            
        # Update global parameters
        epoch_outputs, global_parameters = aggre_method(epoch_outputs)
        global_model.load_state_dict(global_parameters)

        # Evaluate global model on global dataset
        global_evaluator = Evaluator(args, logger)
        global_metric, global_loss = global_evaluator(global_model, dataset.test_dataset)

        # Calculate the profit
        epoch_outputs, publisher_profit, pay_count = epoch_incentive(args, epoch_outputs, global_metric, old_global_metric)
        server_total_profit += publisher_profit

        # Epoch ends
        logger.info("Epoch {} End".format(epoch))

        # Print epoch information
        logger.info("[client]Clients Information:")
        for single_client in epoch_outputs:
            client_dict = single_client.output()
            client_idx = client_dict["idx"]
            for client_key in client_dict:
                if client_key != "idx":
                    logger.info("[client-{}]{}: {}".format(client_idx, client_key, client_dict[client_key]))
        logger.info("[server]Server information:")
        logger.info("[server]loss: {}".format(global_loss))
        logger.info("[server]accuracy: {}".format(global_metric))
        logger.info("[server]epoch_profit: {}".format(publisher_profit / pay_count if pay_count != 0 else 0))
        logger.info("[server]total_profit: {}".format(server_total_profit))

        # Global early stop mechanism
        if args.global_early_stop and epoch > args.no_stoping_epochs:
            global_early_stopping(global_loss, global_model)
            if global_early_stopping.early_stop:
                break

        # Reset some values
        # Only the best performance is meaningful to task participant
        old_global_metric = max(global_metric, old_global_metric)
        epoch_outputs = []

    pickle.dump(local_outputs, open("../clients_data/{}-{}.pickle".format(session_time, args.dataset), "wb"))
    logger.info("Task finished!")