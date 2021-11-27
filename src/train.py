from tqdm import tqdm

from sklearn.model_selection import KFold

import torch
from torch import nn
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch.utils.data import SubsetRandomSampler

from metrics import evaluate, TrainingMetrics


def run_training(
    model: nn.Module, dataset_train: DataLoader, metrics: TrainingMetrics, cv: int = 5,
    lr: float = 1e-2, decay: float = 1e-3, batch_size: int = 4,
    epochs: int = 25, device=None, dense: bool = None,
    verbose: int = 0
):
    
    if cv <= 1:
        
        valid_split = round(len(dataset_train) * (1 if cv == 0 else .8))
        
        # If k for KFold validation is 0, don't use a validation set
        loader_train_args = dict(dataset=dataset_train[:valid_split], batch_size=batch_size)
        loader_train = DenseDataLoader(**loader_train_args) if model.is_dense() \
            else DataLoader(**loader_train_args)
        
        loader_valid_args = dict(dataset=dataset_train[valid_split:], batch_size=batch_size)
        loader_valid = DenseDataLoader(**loader_valid_args) if model.is_dense() \
            else DataLoader(**loader_valid_args)
            
        train(model, loader_train, loader_valid if cv == 1 else None, metrics,
              learning_rate=lr, weight_decay=decay, epochs=epochs, device=device,
              verbose=verbose)
    
    elif cv > 1:
        kfold = KFold(n_splits=cv, shuffle=True)
        
        # Create progress bar
        pbar = tqdm(enumerate(kfold.split(dataset_train)), total=cv, leave=False, disable=(verbose < 0))
        
        for fold, (I_train, I_valid) in pbar:
            if verbose > 0:
                pbar.set_description(
                    f'Fold no. {fold} | Train size: {len(I_train)}, Valid size: {len(I_valid)}')
            
            metrics.set_run(fold)
            
            model.reset()
            
            sampler_train = SubsetRandomSampler(I_train)
            sampler_valid = SubsetRandomSampler(I_valid)
            
            loader_train_args = dict(dataset=dataset_train, batch_size=batch_size, sampler=sampler_train)
            loader_valid_args = dict(dataset=dataset_train, batch_size=batch_size, sampler=sampler_valid)
            
            loader_train = DenseDataLoader(**loader_train_args) if model.is_dense() \
                else DataLoader(**loader_train_args)
            loader_valid = DenseDataLoader(**loader_valid_args) if model.is_dense() \
                else DataLoader(**loader_valid_args)
                    
            train(model, loader_train, loader_valid, metrics,
                  learning_rate=lr, weight_decay=decay, epochs=epochs, device=device,
                  verbose=verbose)
        
    else:
        raise ValueError(f'k={cv} is not appropriate for k-fold cross-validation.')


def train(
    model: nn.Module, loader_train: DataLoader, loader_valid: DataLoader,
    metrics: TrainingMetrics,
    learning_rate: float = 1e-2, weight_decay: float = 1e-3,
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

    # Set progressbar
    pbar = tqdm(range(epochs), disable=(verbose < 0))
    for epoch in pbar:
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
            
        with torch.no_grad():
            acc_train, _ = evaluate(model, loader_train)
            
            if verbose > 0:
                pbar.set_description(f'Epoch {epoch:<3} | Loss {epoch_loss:,.2f} | Acc. {acc_train:,.2f}')
            
            metrics.log_metric('Loss - training', epoch_loss, step=epoch)
            metrics.log_metric('Accuracy - training', acc_train, step=epoch)

            if loader_valid is not None:
                acc_valid, loss_valid = evaluate(model, loader_valid, validation=True)
                
                metrics.log_metric('Loss - validation', loss_valid, step=epoch)
                metrics.log_metric('Accuracy - validation', acc_valid, step=epoch)
