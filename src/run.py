import os
import time
import click
import datetime as dt

import torch
from torch_geometric.loader import DataLoader, DenseDataLoader

import mlflow

from metrics import TrainingMetrics, TestingMetrics
from models.baseline import BaselineGNN
from models.diffpool import DiffPool
from train import run_training
from utils import load_dataset, ASSETS_DIR


@click.command()
@click.option('--model', default='GNN',
              type=click.Choice(['GNN', 'GAT', 'GIN', 'DiffPool'], case_sensitive=False),
              help="Model architecture choice.")
@click.option('--connectivity', default='wasserstein',
              type=click.Choice(['fully', 'organ', 'wasserstein'], case_sensitive=False),
              help="Graph connectivity choice.")
@click.option('--epochs', default=50,
              help="Number of training epochs.")
@click.option('--lr', default=1e-4,
              help="Learning rate.")
@click.option('--decay', default=1e-4,
              help="Optimizer weight decay.")
@click.option('--hidden-dim', default=64,
              help="GNN hidden dimensions.")
@click.option('--batch-size', default=8,
              help="Batch size for training.")
@click.option('--test-size', default=0.2,
              help="Test set size in ratio.")
@click.option('--seed', default=27,
              help="Random seed.")
@click.option('--cv', default=5,
              help="Cross-validation splits.")
@click.option('--distance', default=0.5,
              help="Wasserstein distance threshold for graph creation.")
@click.option('--experiment-name', default='Default',
              help="Assign run to experiment.")
@click.option('--verbose', default=1, type=int,
              help="Print out info for debugging purposes.")
def run(model, connectivity,
        epochs, lr, decay, hidden_dim, batch_size,
        test_size, seed, cv, distance,
        experiment_name, verbose):
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment(experiment_id)
    
    timestamp = dt.datetime.today()
    mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model)
    
    mlflow.log_param('Cross-validation splits', cv)
    mlflow.log_param('Learning Rate', lr)
    mlflow.log_param('Weight Decay', decay)
    mlflow.log_param('Batch Size', batch_size)
    mlflow.log_param('Connectivity', connectivity)
    mlflow.log_param('Hidden dimensions', hidden_dim)
    if connectivity == 'wasserstein':
        mlflow.log_param('Wasserstein threshold', distance)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    model_args = dict(
        num_classes=2,
        hidden_dim=hidden_dim,
        node_features_dim=44)
    
    if model == 'GNN':
        model = BaselineGNN(layer_type='GraphConv', **model_args).to(device)
    elif model == 'GAT':
        model = BaselineGNN(layer_type='GAT', **model_args).to(device)
    elif model == 'DiffPool':
        # 9 is the median amount of nodes in each graph
        # TODO: make this pooling approach more data-driven
        model = DiffPool(**model_args, num_nodes=[9]).to(device)
    elif model == 'GIN':
        model = BaselineGNN(layer_type='GIN', **model_args).to(device)
    else:
        raise ValueError(f'Could not instanciate {model} model')
        
    mlflow.log_param('Model', model)
    mlflow.log_param('Weights', model.param_count())
        
    if verbose > 0:
        print(f"{model} instanciated with {model.param_count()} parameters.\n"
              f"Fetching {connectivity}-connected graph representations and storing "
              f"into DataLoaders with batch size {batch_size}...")
    
    dataset_train, dataset_test = \
        load_dataset(connectivity=connectivity, test_size=test_size, seed=seed,
                     distance=distance, dense=model.is_dense(), verbose=verbose)
    
    start = time.time()
    
    metrics = TrainingMetrics()
    
    run_training(model, dataset_train, 
                 metrics=metrics, cv=cv, lr=lr, decay=decay, batch_size=batch_size,
                 epochs=epochs, dense=model.is_dense(), verbose=verbose)
    
    metrics.send_log()
    
    end = time.time()
    mlflow.log_metric('Elapsed time - training', end - start)
    
    MODEL_PATH = os.path.join(ASSETS_DIR, f'models/{model}-{timestamp}.pkl')
    torch.save(model.state_dict(), MODEL_PATH)
    # mlflow.log_artifact(MODEL_PATH)
    
    loader_test_args = dict(dataset=dataset_test, batch_size=len(dataset_test))
    
    loader_test = DenseDataLoader(**loader_test_args) if model.is_dense() \
        else DataLoader(**loader_test_args)
        
    test_metrics = TestingMetrics(epoch=epochs)
    test_metrics.compute_metrics(model, loader_test)
    test_metrics.send_log(timestamp=timestamp)
    
    mlflow.end_run()
    
    return metrics.get_objective()


if __name__ == '__main__':
    run()
