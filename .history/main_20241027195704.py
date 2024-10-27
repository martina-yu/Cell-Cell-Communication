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

##### KFolf: Cross Validation #####
from sklearn.model_selection import KFold

import optuna
import gymnasium


from ray.tune import CLIReporter
from ray.tune import Trainable
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.train import Result

from utils_yzm import RegularGAT, TrainGAT, train_model, validate_model, test_model, calculate_sparsity
from earlystopping_yzm import EarlyStopping, test_with_early_stopping

import ray
import csv

from ray import train, tune
from ray.tune import ResultGrid

torch.cuda.empty_cache()

# Initialize Ray outside the loop
ray.init()

# Random Seed Setting At the Beginning
set_seed(2024)

# Base file path

# Define constants
brain_data = 'left'
select_family = 'Neurotrophins'

for radius_threshold in range(2400, 3501, 100):
    # Update TrainGAT_path for the current radius_threshold
    TrainGAT_path = f'/home/ /CellCommu/tmp_code/processed_for_CV/Radius_{radius_threshold}_{select_family}_ligand_target_{brain_data}.data'
    exp_name = f"exp_{select_family}_{brain_data}_radius_{radius_threshold}"
    fold_results = []
    results_dfs = []
    for fold in range(5):
    # Update search space
        search_space = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "hidden_channels": tune.choice([64, 128, 256]),
            "heads": 1,
            "dropout": tune.uniform(0.1, 0.6),
            "TrainGAT_path": TrainGAT_path,
            "file_path": file_path,
            "select_family": select_family,
            "brain_data": brain_data,
            "radius_threshold": radius_threshold
        }
        
        # Configure the scheduler and reporter
        scheduler = ASHAScheduler(
            metric="val_loss",  # Ray Tune will optimize for the validation loss
            mode="min",         # Minimize the validation loss
            max_t=1000,          # 1000 epochs
            grace_period=1,     # Minimum number of epochs to run before stopping
            reduction_factor=2  # Controls the rate at which configurations are stopped
        )
        
        reporter = CLIReporter(
            metric_columns=["val_loss", "best_val_loss", "best_epoch", "training_iteration"]
        )
        
        # Use OptunaSearch as the search algorithm
        optuna_search = OptunaSearch(metric="val_loss", mode="min")
        
        # Run the optimization
        tuner = tune.Tuner(
            tune.with_resources(TrainGAT, resources={"cpu": 4, "gpu": 1}),
            run_config=train.RunConfig(
                progress_reporter=reporter,
                checkpoint_config=train.CheckpointConfig(
                    checkpoint_at_end=True, checkpoint_frequency=5
                ),
                name=exp_name,
                storage_path=ray_result_path,
            ),
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=50,  # Number of different hyperparameter configurations to try
                search_alg=optuna_search
            ),
            param_space=search_space
        )
        
        # Fit the tuner (this should be inside the loop)
        results = tuner.fit()
        
        # After fitting, process the results
        experiment_path = os.path.join(ray_result_path, exp_name)
        
        # Restore the tuner for the current experiment
        restored_tuner = tune.Tuner.restore(experiment_path, trainable=TrainGAT)
        result_grid = restored_tuner.get_results()
    
        # Get the results dataframe
        results_df = result_grid.get_dataframe() ## get each best fold's result
    
        #####  TESTING: Save the best parameter ####
        best_result = results.get_best_result("best_val_loss", "min")
        best_params = best_result.config
        torch.save(best_params, f'{file_path}/Radius_{radius_threshold}_{select_family}_{brain_data}_best_params_{fold}.data')
        
        best_val_loss = best_result.metrics["best_val_loss"]
        best_epoch = best_result.metrics["best_epoch"]
        fold_results.append({
            "Radius_Threshold": radius_threshold,
            "Best_Val_Loss": best_val_loss,
            "Best_Epoch": best_epoch,
            "Fold": fold,
            # Store the best configuration as well
        })

    
    #####  Cross-Validation: get the information of the best fold ####
    fold_results = pd.DataFrame(fold_results)
    fold_results.to_csv(f'{file_path}/results_df_{radius_threshold}_{select_family}_{brain_data}.csv')
    
    best_fold_row = fold_results.loc[fold_results['Best_Val_Loss'].idxmin()]
    best_fold = int(best_fold_row['Fold'])
    best_fold_epoch = int(best_fold_row['Best_Epoch'])
        
    #####  TESTING  ####
    #####  TESTING: Save the best val loss to csv file ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(f'/home/ /CellCommu/tmp_code/processed_for_CV/Radius_{radius_threshold}_{select_family}_ligand_target_{brain_data}.data').to(device)
    
    best_params = torch.load(f'{file_path}/Radius_{radius_threshold}_{select_family}_{brain_data}_best_params_{best_fold}.data')
    
    best_model = RegularGAT(
        data.x.shape[1],
        data.labels.shape[1],
        data.y.shape[1],
        best_params['hidden_channels'],
        1, ## head
        best_params['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(best_model.parameters(),lr=best_params["lr"])
    
    basename = f'RegularGAT_best_model_for_Radius_{radius_threshold}_{select_family}_{brain_data}'
    checkpoint = f'{file_path}/best_model/checkpoints/{basename}'  # model weights
    history = f'{file_path}/history/{basename}'  # attention score and weights
    logfile = f'{file_path}/logs/log_{basename}.txt'  # training history

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    if not os.path.exists(history):
        os.makedirs(history)
        
    logfile_dir = os.path.dirname(logfile)
    os.makedirs(logfile_dir, exist_ok=True)

    Path(logfile).touch()
    log_file = open(logfile, 'w')

    train_losses = []
    val_losses = []
    attention_weights_history = [] ## after normalized
    attention_scores_history = []
    
    # start_epoch = 0

    logs = {}
    groups = {'total_loss': ['train_total_loss', 'val_total_loss'],
              'sparsity': ['train_sparsity']}

    for epoch in range(best_result.metrics['best_epoch']):
        # train_total_loss, train_attention_weights, train_attention_scores = train_model(data, device)
        train_total_loss, train_attention_weights, train_attention_scores = train_model(best_model, data, device, optimizer)
        val_total_loss, val_attention_weights, val_attention_scores = validate_model(best_model, data, device)
        sparsity_results = calculate_sparsity(train_attention_weights)
        logs['train_total_loss'] = train_total_loss
        logs['val_total_loss'] = val_total_loss
        logs['train_sparsity'] = sparsity_results['Sparsity']
        log_file.write(f'{logs}\n')

        attention_weights_history.append(train_attention_weights)
        attention_scores_history.append(train_attention_scores)
        
        if (epoch + 1) % 500 == 0:
            torch.save(best_model.state_dict(), f'{checkpoint}/best_model_epoch_{epoch + 1}_{select_family}_{brain_data}.pth')
 
    log_file.close()

    np.save(history + '/attention_weights.npy', np.array(attention_weights_history))
    
    test_loss = test_model(best_model, data, device)

    with open(file_path + f'/RegularGAT_test_losses_{select_family}_{brain_data}.csv', mode='a', newline='') as file:
        writer = csv.writer(file)

        file.seek(0, 2)
        if file.tell() == 0:
            writer.writerow(["Radius_Threshold", "Test_Loss", "Best_Epochs"])

        writer.writerow([radius_threshold, test_loss, best_result.metrics['best_epoch']])
    
    ### Out OF Memory!!!!
    torch.cuda.empty_cache()