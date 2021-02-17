#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class MaliciousChange(object):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

        self.malicious_clients = np.random.choice(range(args.num_users), int(args.num_users * args.malicious_rate), replace=False)

    def data_poision(self, user_groups: dict, source_list: list, dest_list: list):
        for single_client_idx in self.malicious_clients:
            data_idxs = list(user_groups[single_client_idx])
            for single_idx in data_idxs:
                try:
                    source_pos = source_list.index(self.dataset.targets[int(single_idx.item())])
                    self.dataset.targets[int(single_idx.item())] = dest_list[source_pos]
                except ValueError:
                    pass


if __name__ == "__main__":
    # test
    class Args(object):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Args(num_users=10, malicious_rate=0.1)
    dataset = Args(targets=list(range(10)))

    changer = MaliciousChange(args, dataset)
    print(changer.dataset.__dict__)

    user_groups = {i: list(range(10)) for i in range(10)}
    changer.data_poision(user_groups, [1], [3])
    print(dataset.__dict__)

    # print results: source_list: [1, 3], dest_list: [3, 1]
    # {'targets': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
    # {'targets': [0, 3, 2, 1, 4, 5, 6, 7, 8, 9]}
    # print results: source_list: [1], dest_list: [3]
    # {'targets': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
    # {'targets': [0, 3, 2, 3, 4, 5, 6, 7, 8, 9]}