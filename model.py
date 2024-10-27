import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

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

import optuna
import gymnasium
# from .autonotebook import tqdm as notebook_tqdm
from ray import tune
from ray.tune import CLIReporter
from ray.tune import Trainable
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
# from ray.rllib.algorithms.ppo import PPO, PPOConfig

# Define regular GAT model to get initial attention weights
class RegularGAT(torch.nn.Module):
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
        # self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0)
        self.initial_attention_weights_abs = None
        self._initialize_weights()
    
    def forward(self, data):
        x, edge_index, t = data.x, data.edge_index, data.labels
        t_indices = torch.argmax(t, dim=1).to(self.device)
        t_emb = self.embedding(t_indices).to(self.device)  # Get the embedding for T
        x, (edge_index, (attention_scores, attention_weights)) = self.gat1(x, edge_index, return_attention_weights=True)
        print(f'after GAT relu x: {x}')
        
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([x, t_emb], dim=1)
        
        x = self.mlp(x)  # Apply MLP to combined features with LayerNorm
        x = F.dropout(x, p=self.dropout, training=self.training)
        print(f'after mlp and dropout x: {x}')
        
        # x, attention_weights = self.gat2(x, edge_index, return_attention_weights=True)
        
        if self.initial_attention_weights_abs is None:
            self.initial_attention_weights_abs = torch.abs(attention_weights[1]).detach()
        return x.to(self.device), attention_scores.to(self.device), attention_weights.to(self.device)  # Return the attention coefficients as well
   
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                

# Define training process
def train(data, device):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    out, attention_scores, attention_weights = model(data)
    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
    # penalty = model.compute_attention_penalty(attention_weights)
    # total_loss = loss + penalty
    total_loss = loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), attention_weights.cpu().detach().numpy(), attention_scores.cpu().detach().numpy()

def validate(data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out, attention_scores, attention_weights = model(data)
        loss = F.mse_loss(out[data.val_mask], data.y[data.val_mask])
        # penalty = model.compute_attention_penalty(attention_weights)
        # total_loss = loss + penalty
        total_loss = loss
    return total_loss.item(), attention_weights.cpu().detach().numpy(), attention_scores.cpu().detach().numpy()

def test(data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out, _, _ = model(data)
        test_loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask])
    return test_loss.item()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()

data = torch.load(f'{file_path}/processed_for_CV/Radius_1000_Neurotrophins_ligand_target_left.data')

radius_threshold = 1000
model = RegularGAT(
    in_channels_x=data.x.shape[1], 
    in_channels_t=data.labels.shape[1], 
    out_channels=data.y.shape[1], 
    hidden_channels=my_hidden_channels, 
    heads=my_heads, 
    dropout=my_dropout
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=my_lr)

basename = f'RegularGAT_hidden_{my_hidden_channels}_head_{my_heads}_drop_{my_dropout}_lr_{my_lr}_right_Radius_200'

#####

checkpoint = f'{file_path}/rGAT_cosmx_yes_CV/models/checkpoints/{basename}'  # model weights
history = f'{file_path}/rGAT_cosmx_yes_CV/history/{basename}'  # attention score and weights
logfile = f'{file_path}/rGAT_cosmx_yes_CV/logs/log_{basename}.txt'  # training history

if not os.path.exists(checkpoint):
    os.makedirs(checkpoint)
if not os.path.exists(history):
    os.makedirs(history)


log_dir = os.path.dirname(logfile)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
if not os.path.exists(logfile):
    Path(logfile).touch()
    
log_file = open(logfile, 'w')

#####

train_losses = []
val_losses = []
attention_weights_history = [] ## after normalized
attention_scores_history = []

#####

logs = {}

groups = {'total_loss': ['train_total_loss', 'val_total_loss'],
          'sparsity': ['train_sparsity']}

for epoch in range(num_epochs):
    train_total_loss, train_attention_weights, train_attention_scores = train(data, device)
    val_total_loss, val_attention_weights, val_attention_scores = validate(data, device)
    logs['train_total_loss'] = train_total_loss
    logs['val_total_loss'] = val_total_loss
    log_file.write(f'{logs}\n')
    
    attention_weights_history.append(train_attention_weights)
    attention_scores_history.append(train_attention_scores)
    
    if (epoch + 1) % 500 == 0:
        torch.save(model.state_dict(), f'{checkpoint}/model_epoch_{epoch+1}.pth')
        
log_file.close()

np.save(history + '/attention_weights.npy', np.array(attention_weights_history))
test_loss = test(data, device)
print(f'Radius {radius_threshold}: Test MSE: {test_loss:.4f}')

def r_squared(y_true, y_pred):
    """Compute custom r squared.

    Parameters
    ----------
    y_true
        y_true.
    y_pred
        y_pred.

    Returns
    -------
    r2
    """
    print('y_pred is:',y_pred)
    
    y_pred, _ = tf.split(y_pred, num_or_size_splits=2, axis=2)
    # print('after processed y_pred is: \n',y_pred)
    # print('what is splited _: \n',_)
    
    residual = tf.reduce_sum(tf.square(y_true - y_pred))
    # print('residual is? \n',residual)
    
    total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    # print('what is total: \n',total)
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    print(r2)
    return r2