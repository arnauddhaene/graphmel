import os
import click
import time
import datetime as dt

import torch

import mlflow

from metrics import evaluate_accuracy
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
@click.option('--experiment-name', default='Default',
              help="Assign run to experiment.")
@click.option('--verbose', default=2, type=int,
              help="Print out info for debugging purposes.")
def run(model, connectivity,
        epochs, lr, decay, hidden_dim, batch_size,
        test_size, val_size, seed,
        experiment_name, verbose):
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment(experiment_id)
    
    timestamp = dt.datetime.today()
    mlflow.start_run(experiment_id=experiment.experiment_id)
    
    mlflow.log_param('Architecture', model)
    mlflow.log_param('Learning Rate', lr)
    mlflow.log_param('Weight Decay', decay)
    mlflow.log_param('Batch Size', batch_size)
    mlflow.log_param('Connectivity', connectivity)
    mlflow.log_param('Hidden dimensions', hidden_dim)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    model_args = dict(
        num_classes=2,
        hidden_dim=hidden_dim,
        node_features_dim=37)
    
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
        
    if verbose > 1:
        print(f"{model} instanciated with {model.param_count()} parameters.")
        
    mlflow.log_param('Model', model)
    mlflow.log_param('Weights', model.param_count())
    
    if verbose > 1:
        print(f"Creating {connectivity}-connected graph representations and storing "
              f"into DataLoaders with batch size {batch_size}...")
    
    loader_train, loader_valid, loader_test = \
        load_dataset(connectivity=connectivity, batch_size=batch_size,
                     test_size=test_size, val_size=val_size, seed=seed,
                     dense=model.is_dense(), verbose=verbose)
    
    start = time.time()
        
    train(model, loader_train, loader_valid,
          learning_rate=lr, weight_decay=decay, epochs=epochs, device=device,
          verbose=verbose)
    
    end = time.time()
    mlflow.log_metric('Elapsed time - training', end - start)
    
    MODEL_PATH = os.path.join(ASSETS_DIR, f'models/{model}-{timestamp}.pkl')
    torch.save(model.state_dict(), MODEL_PATH)
    # mlflow.log_artifact(MODEL_PATH)
    
    acc_test = evaluate_accuracy(model, loader_test)
    mlflow.log_metric('Accuracy - testing', acc_test, step=epochs)
    
    mlflow.end_run()


if __name__ == '__main__':
    run()
