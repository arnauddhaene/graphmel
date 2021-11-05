import os
import datetime as dt

import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import ASSETS_DIR


def compute_predictions(model: torch.nn.Module, loader: DataLoader, validation: bool = False,
                        device: torch.device = None):
    """Compute accuracy of input model over all samples from the loader.
    
    Args:
        model (torch.nn.Module): NN model
        loader (DataLoader): Data loader to evaluate on
        validation (bool): Validation mode, will also compute loss. Defaults to False.
        device (torch.device), optional:
            Device to use, by default None.
            if None uses cuda if available else cpu.
    Returns:
        List[int]: predictions
        List[int]: ground truth labels
        float: loss. Returns None if not in validation mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    
    if validation:
        criterion = nn.NLLLoss()
        loss = 0.
    
    y_preds = []
    y_trues = []
    
    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.to(device)
            
            out = model(data)
            
            if validation:
                loss += criterion(out, data.y.flatten()).item()

            y_preds.append(out.argmax(dim=1))  # Use the class with highest probability.
            y_trues.append(data.y)  # Check against ground-truth labels.

    y_pred = torch.cat(y_preds).flatten()
    y_true = torch.cat(y_trues).flatten()
    
    return y_true, y_pred, loss if validation else None


def evaluate(model: torch.nn.Module, loader: DataLoader, validation: bool = False,
             device: torch.device = None):
    """Compute accuracy of input model over all samples from the loader.
    
    Args:
        model (torch.nn.Module): NN model
        loader (DataLoader): Data loader to evaluate on
        validation (bool): Validation mode, will also compute loss. Defaults to False.
        device (torch.device), optional:
            Device to use, by default None.
            if None uses cuda if available else cpu.
    Returns:
        float: Accuracy in [0,1]
        float: loss. Returns None if not in validation mode.
    """
    y_true, y_pred, loss = compute_predictions(model, loader, validation, device)

    return torch.sum(y_pred == y_true).item() / len(y_true), loss


class Metrics():
    
    def __init__(self):
        
        self.storage = []
    

class TrainingMetrics(Metrics):

    def __init__(self):
        
        super(TrainingMetrics, self).__init__()
        self.run = 0
    
    def log_metric(self, metric: str, value: float, step: int = 0):
        
        self.storage.append(dict(metric=metric, value=value, step=step, run=self.run))
    
    def incr_run(self):
        
        self.run += 1
    
    def set_run(self, run: int = 0):
        
        self.run = run
        
    def get_objective(self):
        
        df = pd.DataFrame(self.storage)
        
        last_epoch = df.loc[df.groupby('metric')['step'].idxmax()]
        objectives = pd.Series(last_epoch.value.values, index=last_epoch.metric).to_dict()
        
        # return weighted objective of training and validation accuracy
        return .5 * objectives['Accuracy - training'] + .5 * objectives['Accuracy - validation']

    def send_log(self):
    
        df = pd.DataFrame(self.storage)
        
        mean = df.groupby(['metric', 'step']).value.mean().reset_index()
        
        for _, feature in mean.iterrows():
            mlflow.log_metric(feature.metric + ' - avg', feature.value, feature.step)
        
        std = df.groupby(['metric', 'step']).value.std().reset_index()
        
        for _, feature in std.iterrows():
            mlflow.log_metric(feature.metric + ' - std', feature.value, feature.step)


class TestingMetrics(Metrics):
    
    def __init__(self, epoch: int = 0):
        
        super(TestingMetrics, self).__init__()
        self.epoch = epoch
    
    def compute_metrics(self, model: nn.Module, loader: DataLoader):
                
        self.y_true, self.y_pred, _ = compute_predictions(model, loader)
        
        # Add testing accuracy
        self.storage.append(
            dict(metric='Accuracy - testing',
                 value=accuracy_score(self.y_true, self.y_pred))
        )
        
        # Add ROC AUC score
        self.storage.append(
            dict(metric='ROC AUC - testing',
                 value=roc_auc_score(self.y_true, self.y_pred))
        )
        
        # Add other binary classification metrics
        bin_class_metrics = precision_recall_fscore_support(self.y_true, self.y_pred, average='binary')
        for value, metric in zip(list(bin_class_metrics)[:-1], ['precision', 'recall', 'fscore']):
            self.storage.append(dict(metric=(metric.capitalize() + ' - testing'), value=value))
        
    def send_log(self, timestamp: dt.datetime):
        
        for log in self.storage:
            metric, value = tuple(log.values())
            mlflow.log_metric(metric, value, step=self.epoch)
        
        fpath = os.path.join(ASSETS_DIR + 'figures/', f'TEST_METRICS_{timestamp}.png')
        self.plot(fpath)
        mlflow.log_artifact(fpath)
            
    def plot(self, fpath: str):
        
        fig, ax = plt.subplots()
        
        self.plot_confusion_matrix(ax)
        
        clean = [f"{entry['metric'].split(' ')[0]}: {entry['value']:,.3f}" for entry in self.storage]
        
        fig.text(.5, .95, ' | '.join(clean), ha='center', va='center')
        
        plt.savefig(fpath, dpi=200)
            
    def plot_confusion_matrix(self, ax: mpl.axes.Axes) -> mpl.axes.Axes:
        
        mf = pd.DataFrame(confusion_matrix(self.y_true, self.y_pred))
        
        mf.columns, mf.index = ['NPD', 'PD'], ['NPD', 'PD']

        sns.heatmap(mf, annot=True, cmap='Blues', cbar=False, fmt='g', ax=ax)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        
        return ax
