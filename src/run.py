import os
import time
import click
import datetime as dt

import torch
from torch_geometric.loader import DataLoader

import mlflow

from metrics import TrainingMetrics, TestingMetrics
from models.baseline import TimeGNN
from train import run_ensembles
from utils import load_dataset, ASSETS_DIR


@click.command()
@click.option('--model', default='GraphConv',
              type=click.Choice(['GraphConv', 'GAT', 'GIN'], case_sensitive=False),
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
@click.option('--layers', default=10,
              help="GNN hidden layers.")
@click.option('--test-size', default=0.2,
              help="Test set size in ratio.")
@click.option('--val-size', default=0.0,
              help="Validation set size in ratio (of training).")
@click.option('--seed', default=27,
              help="Random seed.")
@click.option('--ensembles', default=5,
              help="Number of ensembles to test.")
@click.option('--suspicious', default=0.5,
              help="Threshold of lesions suspicion for inclusion.")
@click.option('--distance', default=0.5,
              help="Wasserstein distance threshold for graph creation.")
@click.option('--experiment-name', default='Default',
              help="Assign run to experiment.")
@click.option('--run-name', default='',
              help="MLflow run name.")
@click.option('--verbose', default=1, type=int,
              help="Print out info for debugging purposes.")
def run(model, connectivity,
        epochs, lr, decay, hidden_dim, layers,
        test_size, val_size, seed, ensembles, suspicious, distance,
        experiment_name, run_name, verbose):
    
    batch_size = 1
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment(experiment_id)
    
    timestamp = dt.datetime.today()
    mlflow.start_run(experiment_id=experiment.experiment_id, run_name=(model if run_name == '' else run_name))
    
    mlflow.log_param('Ensembles', ensembles)
    mlflow.log_param('Suspicion threshold', suspicious)
    mlflow.log_param('Learning Rate', lr)
    mlflow.log_param('Weight Decay', decay)
    mlflow.log_param('Message Passing Layers', layers)
    mlflow.log_param('Batch Size', batch_size)
    mlflow.log_param('Connectivity', connectivity)
    mlflow.log_param('Hidden dimensions', hidden_dim)
    mlflow.log_param('Test size', test_size)
    mlflow.log_param('Validation size', val_size)
    mlflow.log_param('Epochs', epochs)
    if connectivity == 'wasserstein':
        mlflow.log_param('Wasserstein threshold', distance)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_train, dataset_test = \
        load_dataset(connectivity=connectivity, test_size=test_size, seed=seed,
                     suspicious=suspicious, distance=distance, verbose=verbose)
        
    model_args = dict(
        num_classes=2, hidden_dim=hidden_dim, num_layers=layers,
        lesion_features_dim=dataset_train[0].x.shape[1],
        study_features_dim=dataset_train[0].study_features.shape[1],
        patient_features_dim=dataset_train[0].patient_features.shape[0])
    
    assert ensembles > 0, f'{ensembles} is not appropriate for ensembling.'
    
    if model in ['GraphConv', 'GAT', 'GIN']:
        models = [TimeGNN(layer_type=model, **model_args).to(device) for _ in range(ensembles)]
    else:
        raise ValueError(f'Could not instanciate {model} model')
        
    if verbose > 0:
        print(f"{len(models)} x {models[0]} instanciated with {models[0].param_count()} parameters.\n"
              f"Fetching {connectivity}-connected graph representations and storing "
              f"into DataLoaders with batch size {batch_size}...")
    
    mlflow.log_param('Model type', model)
    mlflow.log_param('Weights', models[0].param_count())
        
    start = time.time()
    
    metrics = TrainingMetrics()
    
    run_ensembles(models, dataset_train, val_size=val_size,
                  metrics=metrics, lr=lr, decay=decay, batch_size=batch_size,
                  epochs=epochs, verbose=verbose)
    
    metrics.send_log(timestamp=timestamp)
    
    end = time.time()
    mlflow.log_metric('Elapsed time - training', end - start)
    
    for i, model in enumerate(models):
        MODEL_PATH = os.path.join(ASSETS_DIR, f'models/{model}-{i}-{timestamp}.pkl')
        torch.save(model.state_dict(), MODEL_PATH)
    
    if len(dataset_test) > 0:
        loader_test_args = dict(dataset=dataset_test, batch_size=batch_size)

        loader_test = DataLoader(**loader_test_args)

        test_metrics = TestingMetrics(epoch=epochs)
        test_metrics.compute_metrics(models, loader_test)
        test_metrics.send_log(timestamp=timestamp)
    
    mlflow.end_run()
    
    return metrics.get_objective()


if __name__ == '__main__':
    run()
