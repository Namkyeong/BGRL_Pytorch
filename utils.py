from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import dropout_adj

from collections import Counter

import os.path as osp
import os

import subprocess
import argparse

import scipy.sparse as sp
import numpy as np

import torch


class Augmentation:

    def __init__(self, p_f1 = 0.2, p_f2 = 0.1, p_e1 = 0.2, p_e2 = 0.3):
        """
        two simple graph augmentation functions --> "Node feature masking" and "Edge masking"
        Random binary node feature mask following Bernoulli distribution with parameter p_f
        Random binary edge mask following Bernoulli distribution with parameter p_e
        """
        self.p_f1 = p_f1
        self.p_f2 = p_f2
        self.p_e1 = p_e1
        self.p_e2 = p_e2
        self.method = "BGRL"
    
    def _feature_masking(self, data):
        feat_mask1 = torch.FloatTensor(data.x.shape[1]).uniform_() > self.p_f1
        feat_mask2 = torch.FloatTensor(data.x.shape[1]).uniform_() > self.p_f2
        x1, x2 = data.x.clone(), data.x.clone()
        x1, x2 = x1 * feat_mask1, x2 * feat_mask2

        edge_index1, edge_attr1 = dropout_adj(data.edge_index, data.edge_attr, p = self.p_e1)
        edge_index2, edge_attr2 = dropout_adj(data.edge_index, data.edge_attr, p = self.p_e2)

        new_data1, new_data2 = data.clone(), data.clone()
        new_data1.x, new_data2.x = x1, x2
        new_data1.edge_index, new_data2.edge_index = edge_index1, edge_index2
        new_data1.edge_attr , new_data2.edge_attr = edge_attr1, edge_attr2

        return new_data1, new_data2

    def __call__(self, data):
        
        return self._feature_masking(data)


"""
The Following code is borrowed from SelfGNN
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="data",
                        help="Path to data directory, where all the datasets will be placed. Default is 'data'")
    parser.add_argument("--name", "-n",type=str, default="cora",
                        help="Name of the dataset. Supported names are: cora, citeseer, pubmed, photo, computers, cs, and physics")
    parser.add_argument("--layers", "-l", nargs="+", default=[
                        256, 128], help="The number of units of each layer of the GNN. Default is [512, 128]")
    parser.add_argument("--pred_hid", '-ph', type=int,
                        default=512, help="The number of hidden units of layer of the predictor. Default is 512")
    parser.add_argument("--init-parts", "-ip", type=int, default=1,
                        help="The number of initial partitions. Default is 1. Applicable for ClusterSelfGNN")
    parser.add_argument("--final-parts", "-fp", type=int, default=1,
                        help="The number of final partitions. Default is 1. Applicable for ClusterSelfGNN")
    parser.add_argument("--aug_params", "-p", nargs="+", default=[
                        0.1, 0.4, 0.4, 0.1], help="Hyperparameters for augmentation (p_f1, p_f2, p_e1, p_e2). Default is [0.2, 0.1, 0.2, 0.3]")
    parser.add_argument("--lr", '-lr', type=float, default=0.0001,
                        help="Learning rate. Default is 0.0001.")
    parser.add_argument("--dropout", "-do", type=float,
                        default=0.0, help="Dropout rate. Default is 0.2")
    parser.add_argument("--cache-step", '-cs', type=int, default=10,
                        help="The step size to cache the model, that is, every cache_step the model is persisted. Default is 100.")
    parser.add_argument("--epochs", '-e', type=int,
                        default=10000, help="The number of epochs")
    parser.add_argument("--device", '-d', type=int,
                        default=2, help="GPU to use")
    return parser.parse_args()


def decide_config(root, name):
    """
    Create a configuration to download datasets
    :param root: A path to a root directory where data will be stored
    :param name: The name of the dataset to be downloaded
    :return: A modified root dir, the name of the dataset class, and parameters associated to the class
    """
    name = name.lower()
    if name == 'cora' or name == 'citeseer' or name == "pubmed":
        root = osp.join(root, "pyg", "planetoid")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Planetoid, "src": "pyg"}
    elif name == "computers":
        name = "Computers"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Amazon, "src": "pyg"}        
    elif name == "photo":
        name = "Photo"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Amazon, "src": "pyg"}
    elif name == "cs" :
        name = "CS"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Coauthor, "src": "pyg"}
    elif name == "physics":
        name = "Physics"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Coauthor, "src": "pyg"}        
    else:
        raise Exception(
            f"Unknown dataset name {name}, name has to be one of the following 'cora', 'citeseer', 'pubmed', 'photo', 'computers', 'cs', 'physics'")
    return params


def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)


def create_masks(data):
    """
    Splits data into training, validation, and test splits in a stratified manner if
    it is not already splitted. Each split is associated with a mask vector, which
    specifies the indices for that split. The data will be modified in-place

    :param data: Data object
    :return: The modified data

    """
    if not hasattr(data, "val_mask"):
        labels = data.y.numpy()
        counter = Counter(labels)
        dev_size = int(labels.shape[0] * 0.1)
        test_size = int(labels.shape[0] * 0.8)

        perm = np.random.permutation(labels.shape[0])
        start = end = 0
        test_labels = []
        dev_labels = []
        for l, c in counter.items():
            frac = c / labels.shape[0]
            ts = int(frac * test_size)
            ds = int(frac * dev_size)
            end += ts
            t_lbl = perm[start:end]
            test_labels.append(t_lbl)
            start = end
            end += ds
            d_lbl = perm[start:end]
            dev_labels.append(d_lbl)
            start = end

        test_index, dev_index = np.concatenate(
            test_labels), np.concatenate(dev_labels)
        data_index = np.arange(labels.shape[0])
        test_mask = torch.tensor(
            np.in1d(data_index, test_index), dtype=torch.bool)
        dev_mask = torch.tensor(
            np.in1d(data_index, dev_index), dtype=torch.bool)
        train_mask = ~(dev_mask + test_mask)
        data.train_mask = train_mask
        data.val_mask = dev_mask
        data.test_mask = test_mask
    return data
