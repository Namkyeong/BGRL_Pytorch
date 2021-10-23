import torch
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected

import os.path as osp

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
        if config["name"] == "WikiCS":
            dataset = DatasetClass(data_dir, transform=T.NormalizeFeatures())
            std, mean = torch.std_mean(dataset.data.x, dim=0, unbiased=False)
            dataset.data.x = (dataset.data.x - mean) / std
            dataset.data.edge_index = to_undirected(dataset.data.edge_index)
        else :
            dataset = DatasetClass(**config["kwargs"], transform=T.NormalizeFeatures())
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
            return [f'byg.data.aug.pt']
        else:
            return [f'byg.data.aug.ip.{self.num_parts}.fp.{self.final_parts}.pt']

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


    def process_full_batch_data(self, data):
        """
        Augmented view data generation using the full-batch data.
        :param view1data:
        :return:
        """
        print("Processing full batch data")

        data = Data(edge_index=data.edge_index, edge_attr= data.edge_attr, 
                    x = data.x, y = data.y, 
                    train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                    num_nodes=data.num_nodes)
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
            data_list = self.process_full_batch_data(data)
            data, slices = self.collate(data_list)
            torch.save((data, slices), processed_path)