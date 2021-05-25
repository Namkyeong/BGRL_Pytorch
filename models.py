from torch_geometric.nn import GCNConv, GATConv, SAGEConv

import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np

from functools import wraps
import copy

torch.manual_seed(0)

"""
The following code is borrowed from BYOL, SelfGNN
and slightly modified for BGRL
"""


class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class Encoder(nn.Module):

    def __init__(self, layer_config, dropout=None, project=False, **kwargs):
        super().__init__()
        rep_dim = layer_config[-1]
        self.dropout = dropout
        self.project = project
        self.prelu = nn.PReLU()
        self.stacked_gnn = nn.ModuleList([GCNConv(layer_config[i-1], layer_config[i]) for i in range(1, len(layer_config))])

    def forward(self, x, edge_index, edge_weight=None):
        for gnn in self.stacked_gnn:
            x = self.prelu(gnn(x, edge_index, edge_weight=edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class BGRL(nn.Module):

    def __init__(self, layer_config, pred_hid, dropout=0.0, moving_average_decay=0.99, epochs=1000, **kwargs):
        super().__init__()
        self.student_encoder = Encoder(layer_config=layer_config, dropout=dropout, **kwargs)
        self.teacher_encoder = None
        self.teacher_ema_updater = EMA(moving_average_decay, epochs)
        rep_dim = layer_config[-1]
        self.student_predictor = nn.Sequential(nn.BatchNorm1d(rep_dim), nn.Linear(rep_dim, pred_hid), nn.PReLU(), nn.BatchNorm1d(pred_hid), nn.Linear(pred_hid, rep_dim))

    @singleton('teacher_encoder')
    def _get_teacher_encoder(self):
        teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(teacher_encoder, False)
        return teacher_encoder

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def encode(self, x, edge_index, edge_weight=None, encoder=None):
        encoder = self.student_encoder if encoder is None else encoder
        encoder.train(self.training)
        return encoder(x, edge_index, edge_weight)

    def forward(self, x1, x2, edge_index_v1, edge_index_v2, edge_weight_v1=None, edge_weight_v2=None):
        v1_student = self.encode(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1)
        v2_student = self.encode(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2)

        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)
        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            v1_teacher = self.encode(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1, encoder=teacher_encoder)
            v2_teacher = self.encode(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2, encoder=teacher_encoder)
            
        loss1 = loss_fn(v1_pred, v2_teacher.detach())
        loss2 = loss_fn(v2_pred, v1_teacher.detach())

        loss = loss1 + loss2
        return v1_student, v2_student, loss.mean()


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        logits = self.linear(x)
        loss = self.cross_entropy(logits, y)
        return logits, loss