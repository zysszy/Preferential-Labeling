import logging

from tqdm import tqdm
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn.functional import embedding

import random

class GraphDataset:
    def __init__(self, filepath):
        max_graph_class = 0
        max_node_class = 0

        with open(filepath) as fin:
            max_num_nodes = 0
            #num_total = 1000
            num_total = int(fin.readline().strip())
            data = [None] * num_total

            for i_graph in tqdm(range(num_total)):
                from_list = []
                to_list = []

                num_nodes = int(fin.readline().strip())
                max_graph_class = 2 #max(max_graph_class, graph_class)
                max_num_nodes = max(max_num_nodes, num_nodes)
                node_classes = [None] * num_nodes
                node_labels = [None] * num_nodes

                for i_node in range(num_nodes):
                    node_label, node_degree, *linked_nodes = map(int, fin.readline().strip().split())
                    node_classes[i_node] = 0
                    node_labels[i_node] = node_label
                    max_node_class = max(max_node_class, 0)
                    assert len(linked_nodes) == node_degree
                    for linked_node in linked_nodes:
                        from_list.append(i_node)
                        to_list.append(linked_node)
                adjacent = torch.tensor([to_list, from_list], dtype=torch.long)
                adjacent_values = torch.ones([adjacent.shape[1]], dtype=torch.float32)
                node_classes = torch.tensor(node_classes, dtype=torch.long)
                node_labels = torch.tensor(node_labels, dtype=torch.long)
                
                num_nodes = torch.tensor(num_nodes, dtype=torch.long)

                data[i_graph] = (num_nodes, node_classes, adjacent, adjacent_values, node_labels)

        self.data = data
        self.num_total = num_total
        self.max_num_nodes = max_num_nodes

        logging.getLogger("DataLoader").info(f"Max Graph Class: {max_graph_class} Max Node Class: {max_node_class} Max Num Nodes: {max_num_nodes}")

    def __len__(self):
        return self.num_total

    def __getitem__(self, idx):
        return self.data[idx]


class DatasetSlice:
    def __init__(self, dataset, slice):
        self.dataset = dataset
        self.slice = slice

    def __len__(self):
        return len(self.slice)

    def __getitem__(self, index):
        return self.dataset[self.slice[index]]


def data_slice(dataset, *, split_idx=0):
    labels = [0 for x in dataset.data]
    random.seed(0)
    np.random.seed(0)
    perm = np.random.permutation(len(labels))
#    return DatasetSlice(dataset, perm[:800]), DatasetSlice(dataset, perm[800:900]), DatasetSlice(dataset, perm[900:])
    return DatasetSlice(dataset, perm[:173751]), DatasetSlice(dataset, perm[173510:193751]), DatasetSlice(dataset, perm[193751:])
    #skf = StratifiedKFold(n_splits=40, shuffle=True, random_state=0)
    #train_idx, test_idx = list(skf.split(np.zeros(len(labels)), labels))[split_idx]
    #return DatasetSlice(dataset, train_idx), DatasetSlice(dataset, test_idx)


class GraphCollateFunction:
    def __init__(self, max_num_nodes):
        self.max_num_nodes = max_num_nodes
        padding_mask_lookup = torch.ones([max_num_nodes, max_num_nodes], dtype=torch.bool)
        padding_mask_lookup = ~torch.triu(padding_mask_lookup)
        self.padding_mask_lookup = padding_mask_lookup

    def __call__(self, data):
        with torch.no_grad():
            batch_size = len(data)
            max_num_nodes = self.max_num_nodes
            padding_mask_lookup = self.padding_mask_lookup

            num_nodes, node_classes, adjacent, adjacent_values, graph_class = zip(*data)

            num_nodes = torch.stack(num_nodes, 0)
            padding_mask = embedding(num_nodes, padding_mask_lookup)

            input_ = torch.zeros([batch_size, max_num_nodes], dtype=torch.long)
            for i, node_class in enumerate(node_classes):
                input_[i, : node_class.shape[0]] = node_class

            num_edges = [x.shape[1] for x in adjacent]
            adjacent = torch.cat(adjacent, dim=1)
            adjacent = torch.cat([torch.empty([1, adjacent.shape[1]]), adjacent], dim=0)
            next_node_id = 0
            for i, num_node in enumerate(num_edges):
                adjacent[0, next_node_id : next_node_id + num_node] = i
                next_node_id = next_node_id + num_node
            adjacent_values = torch.cat(adjacent_values)
            # adjacent = torch.sparse_coo_tensor(adjacent, adjacent_values, [batch_size, max_num_nodes, max_num_nodes])
            l = []
            for k in range(len(graph_class)):
                cla = graph_class[k]
                c = np.zeros(max_num_nodes)
                c[:len(cla)] = cla
                l.append(torch.tensor(c, dtype=torch.long))
                
            labels = torch.stack(l)

            return input_, adjacent, adjacent_values, [batch_size, max_num_nodes, max_num_nodes], num_nodes, padding_mask, labels
