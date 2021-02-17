#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

def metric2reward_func(args, metric):
    return - args.gamma / (math.log(metric, math.e))

def metric2satisfy_func(args, metric):
    return args.phi / (1 + math.e ** (args.alpha - 1 / (1 - args.beta * metric)))

def client_reward(args, local_metric, global_metric, old_global_metric):
    local_nice = local_metric > old_global_metric
    if local_nice:
        local_part = 1 * local_metric
    else:
        local_part = 0

    global_nice = global_metric > old_global_metric
    if global_nice:
        global_part = 1 * global_metric
    else:
        global_part = 0

    if local_nice or global_part:
        local_percent = local_part / (local_part + global_part)
        global_percent = global_part / (local_part + global_part)
    else:
        local_percent = 0
        global_percent = 0

    local_reward = metric2reward_func(args, local_metric)
    global_reward = metric2reward_func(args, global_metric)

    return local_percent * local_reward + global_percent * global_reward, local_nice * local_reward, global_nice * global_reward


def server_satisfy(args, local_metric, global_metric, old_global_metric):
    local_nice = local_metric > old_global_metric
    if local_nice:
        local_part = local_metric * 1
    else:
        local_part = 0

    global_nice = global_metric > old_global_metric
    if global_nice:
        global_part = global_metric * 1
    else:
        global_part = 0

    if local_nice or global_nice:
        local_percent = local_part / (local_part + global_part)
        global_percent = global_part / (local_part + global_part)
    else:
        local_percent = 0
        global_percent = 0

    local_satisfy = metric2satisfy_func(args, local_metric)
    global_satisfy = metric2satisfy_func(args, global_metric)

    return local_percent * local_satisfy + global_percent * global_satisfy

def server_contract_profit(args, local_metric, global_metric, old_global_metric):
    local_reward, local_local, local_global = client_reward(args, local_metric, global_metric, old_global_metric)
    server_profit = server_satisfy(args, local_metric, global_metric, old_global_metric)

    return local_reward, server_profit - local_reward, local_local, local_global

def epoch_incentive(args, local_outputs, global_metric, old_global_metric):
    publisher_profit = 0
    pay_count = 0
    for single_client in local_outputs:
        local_reward, server_profit, local_part, global_part = server_contract_profit(args, single_client.accuracy[-1], global_metric, old_global_metric)

        single_client.update(
            epoch_reward=local_reward,
            sum_rewards=single_client.sum_rewards + local_reward,
            local_part=local_part,
            global_part=global_part)

        if server_profit != 0:
            publisher_profit += server_profit
            pay_count += 1
    
    return local_outputs, publisher_profit, pay_count