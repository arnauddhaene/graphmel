import torch
from torch import nn
from torch.utils.data import DataLoader

import os
import shutil
import datetime as dt
from typing import List

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from mlflow import log_metric


# Update rc parameters
SMALL_SIZE = 10
MEDIUM_SIZE = 12
LARGE_SIZE = 16

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = SMALL_SIZE
mpl.rcParams['axes.titlesize'] = SMALL_SIZE
mpl.rcParams['axes.labelsize'] = MEDIUM_SIZE
mpl.rcParams['xtick.labelsize'] = SMALL_SIZE
mpl.rcParams['ytick.labelsize'] = SMALL_SIZE
mpl.rcParams['legend.fontsize'] = SMALL_SIZE
mpl.rcParams['figure.titlesize'] = LARGE_SIZE

mpl.rcParams['figure.figsize'] = [8.3, 5.1]

mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.facecolor'] = '#F5F5F5'
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['grid.linestyle'] = ':'

FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')

if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)


def clear_figures() -> None:
    """Clear figures directory and all subdirectories"""
    
    for filename in os.listdir(FIGURE_DIR):
        filepath = os.path.join(FIGURE_DIR, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)
    

def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """
    Computes the classification accuracy of a model.
    Args:
        model (nn.Module): model of interest with output (prediction, auxiliary)
        loader (DataLoader): data loader with sample structure
            that follows unpacking (input, target, classes)
    Returns:
        float: classification accuracy
    """
    
    accuracy = 0.
    counter = 0
    
    model.eval()
    
    with torch.no_grad():
        for (input, target, _) in loader:
            output, _ = model(input)
            
            accuracy += (output >= 0.5) == target
            counter += target.size(0)
                
    return (accuracy.sum() / counter).float().item()


def heatmap(values: List[float], ax: mpl.axes.Axes) -> mpl.axes.Axes:
    """
    Plot a seaborn heatmap

    Args:
        values (List[float]): list of values in the following order:
            * true positives
            * false negatives
            * false positives
            * true negatives
    """
    
    mf = pd.DataFrame(
        np.array(values).reshape(2, 2)
    )
        
    mf.columns, mf.index = ['True', 'False'], ['True', 'False']

    sns.heatmap(mf, annot=True, cmap='Blues', fmt='g', ax=ax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    
    return ax


class TrainingMetrics:
    """
    Custom class for tracking and plotting training metrics.
    
    Attributes:
        metrics (dict): dictionary that stores metrics on epoch-key
        current (int): current epoch number stored for printing
        run (int): current run when performing trials
        
    Usage:
        1. Define `metrics = TrainingMetrics()` before getting into epoch-loop
        2. Update metrics with `metrics.add_entry(e, l, a, r)`
        3. print(metrics) to display current metrics
    """
    
    def __init__(self) -> None:
        """Initialize with empty attributes"""
        self.metrics = {}
        self.current = None
        self.run = None
    
    def __repr__(self) -> str:
        """
        Representation method

        Returns:
            str: representation
        """
        return (f"TrainingMetrics instance of size {len(self.metrics)}")
    
    def __str__(self) -> str:
        """
        Print method

        Returns:
            str: to print when `print(self)` is called
        """
        metric = self.metrics[f"R{self.run}E{self.current}"]
        return (f"Epoch {metric['epoch']:02} \t"
                f"Loss {metric['loss']:07.3f} \t"
                f"Accuracy {metric['accuracy'] * 100:06.3f}")
        
    def add_entry(self, epoch: int, loss: float, accuracy: float, run: int = 1) -> None:
        """
        Add entry to metrics

        Args:
            epoch (int): current epoch
            loss (float): loss
            accuracy (float): accuracy
            run (int): the current run number (when doing trials)
        """
        self.run = run
        self.current = epoch
        self.metrics[f"R{run}E{epoch}"] = \
            dict(epoch=epoch, loss=loss, accuracy=accuracy, run=run)
            
        log_metric('Train accuracy', accuracy, step=epoch)
        log_metric('Train loss', loss, step=epoch)
    
    def _average_accuracy(self):
        mf = pd.DataFrame.from_dict(self.metrics, orient='index')
        return mf.accuracy.mean()
    
    def plot(self, directory: str) -> None:
        """Plot metrics

        Args:
            directory (str): sub-directory to save the plots in
        """
        mf = pd.DataFrame.from_dict(self.metrics, orient='index')
        # mf['epoch'] = mf.index
        fig = plt.figure()
        
        ax_loss = fig.add_subplot(111)
        
        ax_loss = sns.lineplot(data=mf, x="epoch", y="loss", label='loss', legend=False,
                               estimator='mean', ci='sd')

        ax_acc = ax_loss.twinx()

        sns.lineplot(data=mf, x="epoch", y="accuracy", label='accuracy', legend=False,
                     color='r', ax=ax_acc,
                     estimator='mean', ci='sd')
        
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1),
                   bbox_transform=ax_loss.transAxes, ncol=2)

        plt.suptitle("Training loss and accuracy")
        
        directory = os.path.join(FIGURE_DIR, directory)

        plt.savefig(os.path.join(directory,
                                 f"TRAINING_METRICS_{dt.datetime.today()}.png"))


class TestMetric():
    """[summary]
    
    Attributes:
        model (torch.nn.Module): model to evaluate
        loader (torch.utils.data.DataLoader): data loader
        confusion (dict): confusion matrix
        accuracy (float): testing accuracy
        precision (float): precision
        recall (float): recall
        f1_score (float): F1 score
    """
    
    def __init__(self, model, data_loader):
        """
        Initiate TestinMetrics instance.

        Args:
            model (nn.Module): model of interest with output (prediction, auxiliary)
            data_loader (DataLoader): data loader with sample structure
                that follows unpacking (input, target, classes)
        """
        self.model = model
        self.loader = data_loader
        self.confusion = dict(true_positive=0., false_negative=0.,
                              false_positive=0., true_negative=0.)
        self.accuracy = 0.
        self.precision = 0.
        self.recall = 0.
        self.f1_score = 0.
        
        self.compute()
        
        log_metric('Test accuracy', self.accuracy)
        log_metric('Precision', self.precision)
        log_metric('Recall', self.recall)
        log_metric('F1-score', self.f1_score)
               
    def compute(self) -> None:
        """Compute different metrics by evaluating the model"""
        
        self.model.eval()
        
        with torch.no_grad():
            for (input, target, _) in self.loader:

                # self.model = self.model.train(False) # TEST @lacoupe
                output, _ = self.model(input)
                
                output = (output >= 0.5)
                
                for out, tar in zip(output, target):
                
                    tar = bool(tar)
                    
                    if out and tar:
                        self.confusion['true_positive'] += 1
                    elif not out and not tar:
                        self.confusion['true_negative'] += 1
                    elif out and not tar:
                        self.confusion['false_positive'] += 1
                    elif not out and tar:
                        self.confusion['false_negative'] += 1
        
        self.accuracy = (self.confusion['true_positive'] + self.confusion['true_negative']) \
            / sum(list(self.confusion.values()))
        
        if (self.confusion['true_positive'] + self.confusion['false_positive']) == 0.:
            self.precision = 0.
        else:
            self.precision = self.confusion['true_positive'] \
                / (self.confusion['true_positive'] + self.confusion['false_positive'])
        
        if (self.confusion['true_positive'] + self.confusion['false_negative']) == 0.:
            self.recall = 0.
        else:
            self.recall = self.confusion['true_positive'] \
                / (self.confusion['true_positive'] + self.confusion['false_negative'])
        
        if (self.precision + self.recall) == 0.:
            self.f1_score = 0.
        else:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        
    def __str__(self) -> str:
        """
        Print method

        Returns:
            str: to print when `print(self)` is called
        """
        return f"Acc. {self.accuracy * 100:06.3f} | Prec. {self.precision * 100:06.3f} | " \
            f"Rec. {self.accuracy * 100:06.3f} | F1 {self.f1_score * 100:06.3f}"
            
    def serialize(self) -> dict:
        """Serialize instance into dictionary

        Returns:
            dict: serialized object
        """
        return {
            'true_positive': self.confusion['true_positive'],
            'false_negative': self.confusion['false_negative'],
            'false_positive': self.confusion['false_positive'],
            'true_negative': self.confusion['true_negative'],
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score
        }
            
    def plot(self, directory: str) -> None:
        """Plot metrics

        Args:
            directory (str): sub-directory to save the plots in
        """
        fig, ax = plt.subplots()
        
        heatmap(list(self.confusion.values()), ax=ax)
        
        plt.suptitle('Test metric: ' + self.__str__())
        
        directory = os.path.join(FIGURE_DIR, directory)

        plt.savefig(os.path.join(directory,
                                 f"TEST_METRIC_{dt.datetime.today()}.png"))


class TestingMetrics():
    
    def __init__(self) -> None:
        """Constructor"""
        self.metrics = []
        self.materialized = None
        
    def add_entry(self, model: nn.Module, loader: DataLoader, time_per_epoch: float,
                  verbose: int) -> None:
        
        test_metric = TestMetric(model, loader)
        self.metrics.append(test_metric)
        
        if (verbose > 0):
            print(f"{test_metric} [{time_per_epoch:0.4f} sec. / epoch]")
            
    def materialize(self):
        self.materialized = list(map(lambda m: m.serialize(), self.metrics))
        
    def save(self):
        if self.materialized is None:
            self.materialize()
        
        mets = pd.DataFrame(self.materialized)
        
        mets.to_csv("testing_metrics.csv", index=False)
    
    def _average_accuracy(self):
        if self.materialized is None:
            self.materialize()
        
        mets = pd.DataFrame(self.materialized)
            
        return mets.accuracy.mean()
    
    def plot(self, directory: str) -> None:
        """Plot metrics

        Args:
            directory (str): sub-directory to save the plots in
        """
        self.materialize()
        
        mets = pd.DataFrame(self.materialized)
        
        fig = plt.figure()
        
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[:, 1])
        
        # Average confusion values
        c_avg = mets[['true_positive', 'false_negative', 'false_positive', 'true_negative']].mean()
        
        if len(mets) == 1:
            c_std = [0., 0., 0., 0.]
        else:
            c_std = \
                mets[['true_positive', 'false_negative', 'false_positive', 'true_negative']].std()
        
        ax1.set_title('Confusion matrix (average)')
        heatmap(list(c_avg), ax=ax1)
        ax2.set_title('Confusion matrix (std. dev.)')
        heatmap(list(c_std), ax=ax2)
        
        scalar_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        scalars = []

        for metric in self.materialized:
            for k, v in metric.items():
                if k in scalar_metrics:
                    scalars.append({'value': v, 'metric': k})
                    
        sf = pd.DataFrame(scalars)
        
        palette = sns.color_palette('Blues', n_colors=4)
        
        ax3.set_title('Scalar metrics')
        sns.barplot(data=sf, x="metric", y="value", ci='sd',
                    palette=palette, ax=ax3)
        ax3.set_ylim(0, 1)
        ax3.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
        
        plt.suptitle('Aggregate test metrics')
        
        plt.tight_layout()
        
        directory = os.path.join(FIGURE_DIR, directory)
        plt.savefig(os.path.join(directory,
                                 f"TESTING_METRICS_{dt.datetime.today()}.png"))
