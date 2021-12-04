import os

import click

import torch
from ray import tune

from utils import ASSETS_DIR, load_dataset
from models.baseline import TimeGNN
from metrics import TrainingMetrics
from train import run_crossval


@click.command()
@click.option('--seed', default=27,
              help="Random seed.")
@click.option('--cv', default=5,
              help="Cross-validation splits.")
@click.option('--suspicious', default=0.5,
              help="Threshold of lesions suspicion for inclusion.")
@click.option('--filename', default='hpopt-results-1',
              help="Threshold of lesions suspicion for inclusion.")
@click.option('--verbose', default=-1, type=int,
              help="Print out info for debugging purposes.")
def tune_hyperparams(seed, cv, suspicious, filename, verbose) -> None:
    
    analysis = tune.run(
        invoke_run,
        config=dict(
            verbose=verbose, cv=cv, seed=seed, suspicious=suspicious,
            epochs=125,
            lr=tune.grid_search([1e-2, 1e-3, 1e-4]),
            decay=tune.grid_search([1e-1, 1e-2, 1e-3]),
            hidden_dim=tune.choice([16, 32, 64]),
            distance=tune.grid_search([5e-1, 1., 2., 5.]),
            layers=tune.choice([5, 10, 15])
        ))
    
    result = analysis.get_best_config(metric='objective', mode='max')
    
    print(f"Best configuration: {result}")
    
    analysis.dataframe().to_csv(os.path.join(ASSETS_DIR, 'results/', filename + '.json'))

    
def invoke_run(config):
    
    verbose, cv, seed, suspicious, epochs, distance, layers = \
        config['verbose'], config['cv'], config['seed'], config['suspicious'], config['epochs'], \
        config['distance'], config['layers']
    
    # Extract hyperparameters
    lr, decay, hidden_dim = config['lr'], config['decay'], config['hidden_dim']
    
    batch_size = 1
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_train, _ = \
        load_dataset(connectivity='wasserstein', seed=seed, test_size=0.,
                     suspicious=suspicious, distance=distance, verbose=verbose)
        
    model_args = dict(
        num_classes=2, hidden_dim=hidden_dim, num_layers=layers,
        lesion_features_dim=dataset_train[0].x.shape[1],
        study_features_dim=dataset_train[0].study_features.shape[1],
        patient_features_dim=dataset_train[0].patient_features.shape[0])
        
    models = TimeGNN(layer_type='GAT', **model_args).to(device)
        
    metrics = TrainingMetrics()
    
    run_crossval(models, dataset_train, metrics=metrics, cv=cv, \
                 lr=lr, decay=decay, batch_size=batch_size, \
                 epochs=epochs, device=device, verbose=verbose)
        
    metric = metrics.get_objective()

    tune.report(objective=metric)


if __name__ == '__main__':
    tune_hyperparams()
