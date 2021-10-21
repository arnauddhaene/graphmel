import os
import click
import time
import datetime as dt

from tqdm import tqdm

import torch
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch.utils.data import SubsetRandomSampler

from sklearn.model_selection import KFold

import mlflow

from metrics import evaluate_accuracy, TrainingMetrics
from models.gnn import GNN
from models.gat import GAT
from models.gin import GIN
from models.diffpool import DiffPool
from train import train
from utils import load_dataset, ASSETS_DIR


@click.command()
@click.option('--model', default='GNN',
              type=click.Choice(['GNN', 'GAT', 'GIN', 'DiffPool'], case_sensitive=False),
              help="Model architecture choice.")
@click.option('--connectivity', default='wasserstein',
              type=click.Choice(['fully', 'organ', 'wasserstein'], case_sensitive=False),
              help="Graph connectivity choice.")
@click.option('--epochs', default=200,
              help="Number of training epochs.")
@click.option('--lr', default=1e-4,
              help="Learning rate.")
@click.option('--decay', default=1e-4,
              help="Optimizer weight decay.")
@click.option('--hidden-dim', default=64,
              help="GNN hidden dimensions.")
@click.option('--batch-size', default=8,
              help="Batch size for training.")
@click.option('--val-size', default=0.2,
              help="Validation set size in ratio.")
@click.option('--test-size', default=0.2,
              help="Test set size in ratio.")
@click.option('--seed', default=21,
              help="Random seed.")
@click.option('--cv', default=5,
              help="Cross-validation splits.")
@click.option('--experiment-name', default='Default',
              help="Assign run to experiment.")
@click.option('--verbose', default=2, type=int,
              help="Print out info for debugging purposes.")
def run(model, connectivity,
        epochs, lr, decay, hidden_dim, batch_size,
        test_size, val_size, seed, cv,
        experiment_name, verbose):
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment(experiment_id)
    
    timestamp = dt.datetime.today()
    mlflow.start_run(experiment_id=experiment.experiment_id)
    
    mlflow.log_param('Cross-validation splits', cv)
    mlflow.log_param('Learning Rate', lr)
    mlflow.log_param('Weight Decay', decay)
    mlflow.log_param('Batch Size', batch_size)
    mlflow.log_param('Connectivity', connectivity)
    mlflow.log_param('Hidden dimensions', hidden_dim)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    model_args = dict(
        num_classes=2,
        hidden_dim=hidden_dim,
        node_features_dim=43)
    
    if model == 'GNN':
        model = GNN(**model_args).to(device)
    elif model == 'GAT':
        model = GAT(**model_args).to(device)
    elif model == 'DiffPool':
        model = DiffPool(**model_args).to(device)
    elif model == 'GIN':
        model = GIN(**model_args).to(device)
    else:
        raise ValueError(f'Could not instanciate {model} model')
        
    mlflow.log_param('Model', model)
    mlflow.log_param('Weights', model.param_count())
        
    if verbose > 1:
        print(f"{model} instanciated with {model.param_count()} parameters.\n"
              f"Creating {connectivity}-connected graph representations and storing "
              f"into DataLoaders with batch size {batch_size}...")
    
    dataset_train, dataset_test = \
        load_dataset(connectivity=connectivity, test_size=test_size, seed=seed,
                     dense=model.is_dense(), verbose=verbose)
    
    start = time.time()
    
    metrics = TrainingMetrics()
    
    kfold = KFold(n_splits=cv, shuffle=True)
    
    for fold, (I_train, I_valid) in tqdm(enumerate(kfold.split(dataset_train)), total=cv):
        
        metrics.set_run(kfold)
        
        model.reset()
        
        if verbose > 1:
            print(f'Fold no. {fold}')
            print(f'Train size: {len(I_train)}, Valid size: {len(I_valid)}')
            print(f'Intersection: {len(list(set(I_train) & set(I_valid)))}')
        
        sampler_train = SubsetRandomSampler(I_train)
        sampler_valid = SubsetRandomSampler(I_valid)
        
        loader_train_args = dict(dataset=dataset_train, batch_size=8, sampler=sampler_train)
        loader_valid_args = dict(dataset=dataset_train, batch_size=8, sampler=sampler_valid)
        
        loader_train = DenseDataLoader(**loader_train_args) if model.is_dense() \
            else DataLoader(**loader_train_args)
        loader_valid = DenseDataLoader(**loader_valid_args) if model.is_dense() \
            else DataLoader(**loader_valid_args)
                
        train(model, loader_train, loader_valid, metrics,
              learning_rate=lr, weight_decay=decay, epochs=epochs, device=device,
              verbose=verbose)
        
    metrics.send_log()
    
    end = time.time()
    mlflow.log_metric('Elapsed time - training', end - start)
    
    MODEL_PATH = os.path.join(ASSETS_DIR, f'models/{model}-{timestamp}.pkl')
    torch.save(model.state_dict(), MODEL_PATH)
    # mlflow.log_artifact(MODEL_PATH)à
    
    loader_test_args = dict(dataset=dataset_test, batch_size=len(dataset_test))
    
    loader_test = DenseDataLoader(**loader_test_args) if model.is_dense() \
        else DataLoader(**loader_test_args)
        
    acc_test = evaluate_accuracy(model, loader_test)
    
    mlflow.log_metric('Accuracy - testing', acc_test, step=epochs)
    mlflow.end_run()


if __name__ == '__main__':
    run()
