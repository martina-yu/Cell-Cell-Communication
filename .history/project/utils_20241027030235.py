import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, degree, add_self_loops, remove_self_loops, softmax, is_torch_sparse_tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor, OptPairTensor, Size, NoneType

from torch_sparse import SparseTensor, set_diag
import time

from torch_scatter import scatter_add
from typing import Optional, Tuple, Union
from livelossplot import PlotLosses
from pathlib import Path
from utility import MyGATConv, set_seed

### cross-validation ###
from sklearn.model_selection import KFold

import optuna
import gymnasium
# from .autonotebook import tqdm as notebook_tqdm
from ray import tune
from ray.tune import CLIReporter
from ray.tune import Trainable
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
# from ray.rllib.algorithms.ppo import PPO, PPOConfig

import ray
import csv


## Define regular GAT model to get initial attention weights

class RegularGAT(torch.nn.Module):
    # def __init__(self, in_channels_x, in_channels_t, out_channels, hidden_channels, heads, dropout, num_epochs):
    def __init__(self, in_channels_x, in_channels_t, out_channels, hidden_channels, heads, dropout):
        super(RegularGAT, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = torch.nn.Embedding(in_channels_t, hidden_channels).to(self.device)  # Embedding layer for T: cell_types
        self.gat1 = MyGATConv(in_channels_x, hidden_channels, heads=heads, dropout=0, add_self_loops=False)  # default dropout=0
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_channels), 
            torch.nn.Linear(hidden_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(out_channels)
        ).to(self.device)
        self.bn = torch.nn.BatchNorm1d(out_channels * heads).to(self.device)
        self.dropout = dropout
        # self.num_epochs = num_epochs
        # self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0)
        self.initial_attention_weights_abs = None
        self._initialize_weights()
    
    def forward(self, data):
        x, edge_index, t = data.x, data.edge_index, data.labels
        t_indices = torch.argmax(t, dim=1).to(self.device)
        t_emb = self.embedding(t_indices).to(self.device)  # Get the embedding for T
        x, (edge_index, (attention_scores, attention_weights)) = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # print('before concatenate:',x.size())
        # print('t_emb size before concatenate:',t_emb.size())
        x = torch.cat([x, t_emb], dim=1)
        # print('x size after concatenate:',x.size())# Concatenate X and embedded T
        x = self.mlp(x)  # Apply MLP to combined features with LayerNorm
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x, attention_weights = self.gat2(x, edge_index, return_attention_weights=True)
        if self.initial_attention_weights_abs is None:
            self.initial_attention_weights_abs = torch.abs(attention_weights[1]).detach()
        return x.to(self.device), attention_scores.to(self.device), attention_weights.to(self.device)  # Return the attention coefficients as well
   
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)


class TrainGAT(tune.Trainable):
    def setup(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.TrainGAT_path = config["TrainGAT_path"]
        self.file_path = config["file_path"]
        self.select_family = config["select_family"]
        self.radius_threshold = config["radius_threshold"]
        self.brain_data = config["brain_data"]
        self.data = torch.load(self.TrainGAT_path).to(self.device) ## .data file
        self.data.x = self.data.x.to(self.device)
        self.data.y = self.data.y.to(self.device)

        ##### KFolf: Cross Validation #####
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        self.folds = list(self.kfold.split(self.data.x))

        self.fold_idx = 0
        self.set_fold_data(self.fold_idx)
        ##### KFolf: Cross Validation #####
        
        self.model = RegularGAT(
            self.data.x.shape[1],
            self.data.labels.shape[1],
            self.data.y.shape[1],
            config["hidden_channels"],
            config["heads"],
            config["dropout"]
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
        # Track best validation loss
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        ##### KFolf: Cross Validation #####

    ##### KFolf: Cross Validation #####
    def set_fold_data(self, fold_idx):
        """Sets the training and validation masks for the current fold."""
        train_idx, val_idx = self.folds[fold_idx]
        self.data.train_mask = torch.zeros(self.data.x.size(0), dtype=torch.bool).to(self.device)
        self.data.val_mask = torch.zeros(self.data.x.size(0), dtype=torch.bool).to(self.device)
        self.data.train_mask[train_idx] = True
        self.data.val_mask[val_idx] = True

    def step(self):
        # Train the model for one epoch
        self.model.train()
        self.optimizer.zero_grad()
        out, attention_scores, attention_weights = self.model(self.data)
        loss = F.mse_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        # Validation step
        self.model.eval()
        with torch.no_grad():
            val_out, _, _ = self.model(self.data)
            val_loss = F.mse_loss(val_out[self.data.val_mask], self.data.y[self.data.val_mask])
        
        # Track the best validation loss
        if val_loss.item() < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self.best_epoch = self.iteration  # Store the current epoch as the best

        # Return validation loss for Ray Tune to track
        return {"val_loss": val_loss.item(), "best_val_loss": self.best_val_loss, "best_epoch": self.best_epoch}

    ##### KFolf: Cross Validation #####
    def next_fold(self):
        """Move to the next fold for cross-validation."""
        self.fold_idx += 1
        if self.fold_idx < len(self.folds):
            self.set_fold_data(self.fold_idx)
            self.model = RegularGAT(
                self.data.x.shape[1],
                self.data.labels.shape[1],
                self.data.y.shape[1],
                self.config["hidden_channels"],
                self.config["heads"],
                self.config["dropout"]
            ).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def get_best_fold(self):
        """Return the index of the best fold based on validation loss."""
        return self.best_fold_idx
    
    ##### KFolf: Cross Validation #####        
    def evaluate(self):
        """Evaluate the best fold on the test data."""
        # Set the data for the best fold for evaluation
        self.set_fold_data(self.best_fold_idx)
        self.model.eval()
        with torch.no_grad():
            out, _, _ = self.model(self.data)
            test_loss = F.mse_loss(out[self.data.test_mask], self.data.y[self.data.test_mask])
        return {"test_loss": test_loss.item()}
    
    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save(self.model.state_dict(), path)
        return checkpoint_dir
    
    def load_checkpoint(self, checkpoint_path):
        # self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, "checkpoint")))
    
    # def evaluate(self):
    #     self.model.eval()
    #     with torch.no_grad():
    #         out, _, _ = self.model(self.data)
    #         test_loss = F.mse_loss(out[self.data.test_mask], self.data.y[self.data.test_mask])
    #     return {"test_loss": test_loss.item()}

# def train_model(data, device):
#     best_model.train()
#     data = data.to(device)
#     optimizer.zero_grad()
#     out, attention_scores, attention_weights = best_model(data)
#     loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
#     # penalty = model.compute_attention_penalty(attention_weights)
#     # total_loss = loss + penalty
#     total_loss = loss
#     total_loss.backward()
#     optimizer.step()
#     return total_loss.item(), attention_weights.cpu().detach().numpy(), attention_scores.cpu().detach().numpy()

# Original file where train_model is defined

def train_model(best_model, data, device, optimizer):
    best_model.train()
    data = data.to(device)
    optimizer.zero_grad()
    out, attention_scores, attention_weights = best_model(data)
    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
    # penalty = model.compute_attention_penalty(attention_weights)
    # total_loss = loss + penalty
    total_loss = loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), attention_weights.cpu().detach().numpy(), attention_scores.cpu().detach().numpy()

def validate_model(best_model, data, device):
    best_model.eval()
    data = data.to(device)
    with torch.no_grad():
        out, attention_scores, attention_weights = best_model(data)
        loss = F.mse_loss(out[data.val_mask], data.y[data.val_mask])
        # penalty = model.compute_attention_penalty(attention_weights)
        # total_loss = loss + penalty
        total_loss = loss
    return total_loss.item(), attention_weights.cpu().detach().numpy(), attention_scores.cpu().detach().numpy()

def test_model(best_model, data, device):
    best_model.eval()
    data = data.to(device)
    with torch.no_grad():
        out, _, _ = best_model(data)
        test_loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask])
    return test_loss.item()
    
# def test_model(data, device):
#     best_model.eval()
#     data = data.to(device)
#     with torch.no_grad():
#         out, _, _ = best_model(data)
#         test_loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask])
#     return test_loss.item()

def calculate_sparsity(data):

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    num_zeros = np.sum(data == 0)
    total_elements = data.size
    sparsity = num_zeros / total_elements
    density = 1 - sparsity
    l0_norm = np.sum(data != 0)

    if l0_norm > 0:
        mean_non_zero = np.mean(data[data != 0])
    else:
        mean_non_zero = 0

    return {
        "Sparsity": sparsity,
        "Density": density,
        "L0 Norm": l0_norm,
        "Mean of Non-Zero Elements": mean_non_zero
    }
