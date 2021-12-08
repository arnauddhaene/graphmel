from tqdm import tqdm

from collections import Counter

from sklearn.model_selection import KFold

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler, random_split

from metrics import evaluate, TrainingMetrics


def run_ensembles(
    models: nn.ModuleList, dataset_train: DataLoader, metrics: TrainingMetrics,
    lr: float = 1e-2, decay: float = 1e-3, batch_size: int = 4, val_size: float = 0.0,
    epochs: int = 25, device=None, verbose: int = 0
):
            
    # Create progress bar
    pbar = tqdm(enumerate(models), total=len(models), leave=False, disable=(verbose < 0))
    
    train_len = round(len(dataset_train) * (1 - val_size))
    lengths = [train_len, len(dataset_train) - train_len]
    
    for ensemble, model in pbar:
        if val_size > 0.:
            I_train, I_valid = random_split(range(len(dataset_train)), lengths=lengths,
                                            generator=torch.Generator().manual_seed(42 + ensemble))
            
            sampler_valid = SubsetRandomSampler(I_valid)
            loader_valid_args = dict(dataset=dataset_train, batch_size=batch_size, sampler=sampler_valid)
            loader_valid = DataLoader(**loader_valid_args)
        else:
            I_train, I_valid = range(len(dataset_train)), []
            loader_valid = None
            
            model.reset()
            
        if verbose > 0:
            pbar.set_description(
                f'Ensemble no. {ensemble} | Train size: {len(I_train)}, Valid size: {len(I_valid)}')
        
        metrics.set_run(ensemble)
        
        sampler_train = SubsetRandomSampler(I_train)
        loader_train_args = dict(dataset=dataset_train, batch_size=batch_size, sampler=sampler_train)
        loader_train = DataLoader(**loader_train_args)
                
        train(model, loader_train, loader_valid, metrics,
              learning_rate=lr, weight_decay=decay, epochs=epochs, device=device,
              verbose=verbose)


def run_crossval(
    model: nn.Module, dataset_train: DataLoader, metrics: TrainingMetrics, cv: int = 5,
    lr: float = 1e-2, decay: float = 1e-3, batch_size: int = 1,
    epochs: int = 25, device=None, verbose: int = 0
):
        
    if cv > 1:
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
            
            loader_train = DataLoader(**loader_train_args)
            loader_valid = DataLoader(**loader_valid_args)
                    
            train(model, loader_train, loader_valid, metrics,
                  learning_rate=lr, weight_decay=decay, epochs=epochs, device=device,
                  verbose=verbose)
        
    else:
        raise ValueError(f'k={cv} is not appropriate for k-fold cross-validation.')


def train(
    model: nn.Module, loader_train: DataLoader, loader_valid: DataLoader,
    metrics: TrainingMetrics,
    learning_rate: float = 1e-2, weight_decay: float = 1e-3, gamma: float = .2,
    epochs: int = 25, device=None, verbose: int = 0
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
        verbose (int, optional): print info. Defaults to 0.
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    class_weights = list(Counter(map(lambda d: d.y.item(), loader_train.dataset)).values())
    class_weights = torch.tensor(class_weights, dtype=torch.double).div(len(loader_train.dataset))
    
    criterion = nn.NLLLoss(weight=class_weights)
    aux_criterion = nn.NLLLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Set progressbar
    pbar = tqdm(range(round(epochs)), disable=(verbose < 0))
    for epoch in pbar:
        epoch_loss = 0.

        model.train()
        
        for data in loader_train:
            data.to(device)
            
            optimizer.zero_grad()
    
            output, aux = model(data)
            loss = criterion(output, data.y.flatten())
            aux_loss = aux_criterion(aux, data.aux_y[0, :].long())
            
            combined_loss = loss + gamma * aux_loss
            combined_loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss per datapoint to be able to compare to validation loss
        epoch_loss /= len(loader_train.dataset)
            
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
