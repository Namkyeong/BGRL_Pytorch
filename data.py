from torch_geometric.data import Data, ClusterData, InMemoryDataset
from torch_geometric.utils import subgraph

import numpy as np
import torch.nn.functional as F
import torch

import os.path as osp
import sys

import utils


def download_pyg_data(config):
    """
    Downloads a dataset from the PyTorch Geometric library

    :param config: A dict containing info on the dataset to be downloaded
    :return: A tuple containing (root directory, dataset name, data directory)
    """
    leaf_dir = config["kwargs"]["root"].split("/")[-1].strip()
    data_dir = osp.join(config["kwargs"]["root"], "" if config["name"] == leaf_dir else config["name"])
    dst_path = osp.join(data_dir, "raw", "data.pt")
    if not osp.exists(dst_path):
        DatasetClass = config["class"]
        dataset = DatasetClass(**config["kwargs"])
        utils.create_masks(data=dataset.data)
        torch.save((dataset.data, dataset.slices), dst_path)
    return config["kwargs"]["root"], config["name"], data_dir


def download_data(root, name):
    """
    Download data from different repositories. Currently only PyTorch Geometric is supported

    :param root: The root directory of the dataset
    :param name: The name of the dataset
    :return:
    """
    config = utils.decide_config(root=root, name=name)
    if config["src"] == "pyg":
        return download_pyg_data(config)


class Dataset(InMemoryDataset):

    """
    A PyTorch InMemoryDataset to build multi-view dataset through graph data augmentation
    """

    def __init__(self, root="data", name='cora', num_parts=1, final_parts=1, augumentation=None, transform=None,
                 pre_transform=None):
        self.num_parts = num_parts
        self.final_parts = final_parts
        self.augumentation = augumentation
        self.root, self.name, self.data_dir = download_data(root=root, name=name)
        utils.create_dirs(self.dirs)
        super().__init__(root=self.data_dir, transform=transform, pre_transform=pre_transform)
        path = osp.join(self.data_dir, "processed", self.processed_file_names[0])
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ["data.pt"]

    @property
    def processed_file_names(self):
        if self.num_parts == 1:
            return [f'byg.data.aug.{self.augumentation.method}.pt']
        else:
            return [f'byg.data.aug.{self.augumentation.method}.ip.{self.num_parts}.fp.{self.final_parts}.pt']

    @property
    def raw_dir(self):
        return osp.join(self.data_dir, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.data_dir, "processed")

    @property
    def model_dir(self):
        return osp.join(self.data_dir, "model")

    @property
    def result_dir(self):
        return osp.join(self.data_dir, "result")

    @property
    def dirs(self):
        return [self.raw_dir, self.processed_dir, self.model_dir, self.result_dir]

    def process_cluster_data(self, data):
        """
        Augmented view data generation based on clustering.

        :param data:
        :return:
        """
        data_list = []
        clusters = []
        num_parts, cluster_size = self.num_parts, self.num_parts // self.final_parts

        # Cluster the data
        cd = ClusterData(data, num_parts=num_parts)
        for i in range(1, cd.partptr.shape[0]):
            cls_nodes = cd.perm[cd.partptr[i - 1]: cd.partptr[i]]
            clusters.append(cls_nodes)

        # Randomly merge clusters and apply transformation
        np.random.shuffle(clusters)
        for i in range(0, len(clusters), cluster_size):
            end = i + cluster_size if len(clusters) - i > cluster_size else len(clusters)
            cls_nodes = torch.cat(clusters[i:end]).unique()
            sys.stdout.write(f'\rProcessing cluster {i + 1}/{len(clusters)} with {self.final_parts} nodes')
            sys.stdout.flush()

            x = data.x[cls_nodes]
            y = data.y[cls_nodes]
            train_mask = data.train_mask[cls_nodes]
            dev_mask = data.val_mask[cls_nodes]
            test_mask = data.test_mask[cls_nodes]
            edge_index, edge_attr = subgraph(cls_nodes, data.edge_index, relabel_nodes=True)
            data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=cls_nodes.shape[0])
            view1data, view2data = self.augumentation(data)
            if not hasattr(view1data, "edge_attr") or view1data.edge_attr is None:
                view1data.edge_attr = torch.ones(view1data.edge_index.shape[1])
            if not hasattr(view2data, "edge_attr") or view2data.edge_attr is None:
                view2data.edge_attr = torch.ones(view2data.edge_index.shape[1])
            diff = abs(view2data.x.shape[1] - view1data.x.shape[1])
            if diff > 0:
                smaller_data = view1data if view1data.x.shape[1] < view2data.x.shape[1] else view2data
                smaller_data.x = F.pad(smaller_data.x, pad=(0, diff))
                view1data.x = F.normalize(view1data.x)
                view2data.x = F.normalize(view2data.x)
            print(view1data)
            print(view2data)
            new_data = Data(y=y, x1=view1data.x, x2=view2data.x, 
                            edge_index1=view1data.edge_index, edge_index2=view2data.edge_index,
                            edge_attr1=view1data.edge_attr, edge_attr2=view2data.edge_attr, 
                            test_x = x, test_edge_index = edge_index, test_edge_attr = edge_attr,
                            train_mask=train_mask, dev_mask=dev_mask, test_mask=test_mask, 
                            num_nodes=cls_nodes.shape[0], nodes=cls_nodes)
            data_list.append(new_data)
        print()
        return data_list

    def process_full_batch_data(self, data):
        """
        Augmented view data generation using the full-batch data.

        :param view1data:
        :return:
        """
        print("Processing full batch data")
        view1data, view2data = self.augumentation(data)
        diff = abs(view2data.x.shape[1] - view1data.x.shape[1])
        if diff > 0:
            """
            Data augmentation on the features could lead to mismatch between the shape of the two views,
            hence the smaller view should be padded with zero. (smaller_data is a reference, changes will
            reflect on the original data)
            """
            smaller_data = view1data if view1data.x.shape[1] < view2data.x.shape[1] else view2data
            smaller_data.x = F.pad(smaller_data.x, pad=(0, diff))
            view1data.x = F.normalize(view1data.x)
            view2data.x = F.normalize(view2data.x)
        print(view1data)
        print(view2data)
        nodes = torch.tensor(np.arange(view1data.num_nodes), dtype=torch.long)
        data = Data(nodes=nodes, edge_index1=view1data.edge_index, edge_index2=view2data.edge_index,
                    edge_attr1=view1data.edge_attr, edge_attr2=view2data.edge_attr, 
                    x1=view1data.x, x2=view2data.x, y=view1data.y,
                    test_x = data.x, test_edge_index = data.edge_index, test_edge_attr = data.edge_attr,
                    train_mask=view1data.train_mask, dev_mask=view1data.val_mask, test_mask=view1data.test_mask, 
                    num_nodes=view1data.num_nodes)
        return [data]

    def download(self):
        pass

    def process(self):
        """
        Process either a full batch or cluster data.

        :return:
        """
        processed_path = osp.join(self.processed_dir, self.processed_file_names[0])
        if not osp.exists(processed_path):
            path = osp.join(self.raw_dir, self.raw_file_names[0])
            data, _ = torch.load(path)
            edge_attr = data.edge_attr
            edge_attr = torch.ones(data.edge_index.shape[1]) if edge_attr is None else edge_attr
            data.edge_attr = edge_attr

            if self.num_parts == 1:
                data_list = self.process_full_batch_data(data)
            else:
                data_list = self.process_cluster_data(data)
            data, slices = self.collate(data_list)
            torch.save((data, slices), processed_path)
