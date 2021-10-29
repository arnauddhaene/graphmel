from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics import evaluate, TrainingMetrics


def train(
    model: nn.Module, loader_train: DataLoader, loader_valid: DataLoader,
    metrics: TrainingMetrics,
    learning_rate: float = 1e-2, weight_decay: float = 1e-3, gamma: float = .5,
    epochs: int = 25, device=None, dense: bool = None,
    verbose: int = 0
) -> None:
    """
    Train model

    Args:
        model (nn.Module): model
        loader_train (DataLoader): data loader
        loader_valid (DataLoader): data loader for validation
        metrics (metrics.TrainingMetrics): metrics object to store results in
        learning_rate (float, optional): learning rate. Defaults to 1e-2.
        weight_decay (float, optional): weight decay for Adam. Defaults to 1e-3.
        epochs (int, optional): number of epochs. Defaults to 25.
        dense (bool), optional:
            train model using dense representation, by default None.
            if None, `model.is_dense()` is called
        verbose (int, optional): print info. Defaults to 0.
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in tqdm(range(epochs), disable=(verbose < 0)):
        epoch_loss = 0.

        model.train()
        
        for data in loader_train:
            data.to(device)
            
            output = model(data)
            
            loss = criterion(output, data.y.flatten())
    
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            loss.backward()
            
            optimizer.step()
            
        acc_train, _ = evaluate(model, loader_train)
        acc_valid, loss_valid = evaluate(model, loader_valid, validation=True)
            
        with torch.no_grad():
            metrics.log_metric('Loss - training', epoch_loss, step=epoch)
            metrics.log_metric('Loss - validation', loss_valid, step=epoch)
            metrics.log_metric('Accuracy - training', acc_train, step=epoch)
            metrics.log_metric('Accuracy - validation', acc_valid, step=epoch)
