from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

import mlflow

from metrics import evaluate_accuracy


def train(
    model: nn.Module, loader_train: DataLoader, loader_valid: DataLoader,
    learning_rate: float = 1e-2, weight_decay: float = 1e-3, gamma: float = .5,
    epochs: int = 25, device=None, dense: bool = False,
    verbose: int = 0
) -> None:
    """
    Train model

    Args:
        model (nn.Module): model
        loader_train (DataLoader): data loader
        loader_valid (DataLoader): data loader for validation
        learning_rate (float, optional): learning rate. Defaults to 1e-2.
        weight_decay (float, optional): weight decay for Adam. Defaults to 1e-3.
        epochs (int, optional): number of epochs. Defaults to 25.
        metrics (metrics.TrainingMetrics): metrics object to store results in
        dense (bool, optional): train model using dense representation
        verbose (int, optional): print info. Defaults to 0.

    Returns:
        TrainingMetrics: metrics of training run
    """

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.

        model.train()
        
        for batch in tqdm(loader_train, leave=False):
            batch.to(device)
            
            if dense:
                output, _, _ = model(batch.x, batch.adj, batch.mask)
            else:
                output = model(batch.x, batch.edge_index, batch.batch)
            
            loss = criterion(output, batch.y.flatten())
    
            optimizer.zero_grad()
            
            loss.backward()
            epoch_loss += batch.y.size(0) * loss.item()
            
            optimizer.step()
            
        acc_train = evaluate_accuracy(model, loader_train)
        acc_valid = evaluate_accuracy(model, loader_valid)
            
        with torch.no_grad():
            mlflow.log_metric('Loss - training', epoch_loss, step=epoch)
            mlflow.log_metric('Accuracy - training', acc_train, step=epoch)
            mlflow.log_metric('Accuracy - validation', acc_valid, step=epoch)
