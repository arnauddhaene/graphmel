import os

from typing import List

from itertools import permutations

import datetime as dt

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer

import mlflow

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
DATA_DIR = os.path.join(BASE_DIR, 'data/')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets/')

CONNECTION_DIR = '/Volumes/lts4-immuno/'
DATA_FOLDERS = ['data_2021-09-20', 'data_2021-10-04']

FILES = dict(
    lesions='melanoma_lesion-info_organ-overlap_2021-09-17_anonymized_cleaned_all.csv',
    lesion_mapping='melanoma_lesion_mapping_2021-09-20_anonymized.csv',
    patients='melanoma_patient-level_summary_anonymized.csv',
    studies='melanoma_study_level_summary_anonymized.csv',
    blood='melanoma_info_patient-treatment-blood-mutation_2021-10-04_anonymized.csv',
    progression='melanoma_petct-exams_progression-status_2021-10-04_anonymized.csv')


def create_dataset(
    connectivity: str = 'organ',
    filepath: str = '', timestamp: dt.datetime = dt.datetime.today(),
    verbose: int = 0
):
    # Fetch lesions datafile
    lesions = pd.read_csv(os.path.join(filepath, FILES['lesions_file']))
    lesions['study_phase'] = lesions.study_name.apply(extract_study_phase)
    
    # Compute TMTV by summing `vol_ccm` per study
    labels = lesions.groupby(['gpcr_id', 'study_phase']) \
        .vol_ccm.sum().to_frame('tmtv').reset_index()

    # Find id of each patient's baseline TMTV in `labels` DataFrame
    baseline_idx = labels.groupby('gpcr_id').study_phase.idxmin().to_dict()
    # Assign baseline TMTV value to each study
    labels['baseline'] = labels.gpcr_id.apply(
        lambda i: labels.loc[baseline_idx[i]].tmtv)
    # Classify response following the TMTV value of each study
    labels['response'] = labels.apply(classify_response, axis=1)
    
    # Creating post-1 dataset
    # Add classified response to each lesion
    df = lesions[lesions.study_phase == 1] \
        .merge(labels[labels.study_phase == 1][['gpcr_id', 'response']], on='gpcr_id')
    # Filter out baseline scans and unwanted features
    df = df[df.response != 'baseline']
    unwanted_features = ['study_name', 'roi_id', 'roi_name', 'lesion_label_id', 'study_phase']
    df = df[[feature for feature in list(df.columns) if feature not in unwanted_features]]

    if verbose > 0:
        print(f'Total patients with non-baseline post-1 studies: {len(df.gpcr_id.unique())}')
    
    x_features = list(df.columns)
    x_features.remove('response')
    x_features.remove('gpcr_id')
    x_features.remove('is_malignant')

    x_features_categorical = ['pars_bodypart_petct', 'pars_region_petct',
                              'pars_subregion_petct', 'pars_laterality_petct',
                              'pars_classification_petct', 'assigned_organ']
    x_features_numerical = ['vol_ccm', 'max_suv_val', 'mean_suv_val',
                            'min_suv_val', 'sd_suv_val']

    # Create one-hot encoder and standard scaler for node-level features
    encoder = OneHotEncoder(drop='if_binary').fit(df[x_features_categorical])
    scaler = StandardScaler().fit(df[x_features_numerical])

    dataset = []

    for patient in list(df.gpcr_id.unique()):

        # Create patient sub-DataFrame of all his post-1 study lesions
        pdf = df[df.gpcr_id == int(patient)].reset_index()

        num_nodes = pdf.shape[0]
        edge_index = []

        # Connect lesions using different rhetorics
        if connectivity == 'organ':
            for i in range(num_nodes):
                source = pdf.loc[i].assigned_organ
                targets = list(pdf[pdf.assigned_organ == source].index)

                edge_index.extend([[i, j] for j in targets if i != j])
        elif connectivity == 'fully':
            edge_index = list(permutations(range(num_nodes), 2, ))
        else:
            raise ValueError(f'Connectivity value not accepted: {connectivity}.'
                             "Must be either 'fully' or 'organ'.")

        edge_index = torch.tensor(edge_index).t()

        x = torch.tensor(np.concatenate((
            encoder.transform(pdf[x_features_categorical]).todense(),
            scaler.transform(pdf[x_features_numerical])
        ), axis=1).astype(np.float32))

        y = pdf[pdf.response == 'response'].shape[0] / pdf.shape[0]

        dataset.append(Data(x=x, edge_index=edge_index, num_nodes=num_nodes, y=int(y)))
        
    DATASET_PATH = os.path.join(DATA_DIR, f'checkpoints/dataset-{connectivity}.pt')
    torch.save(dataset, DATASET_PATH)
    mlflow.log_artifact(DATASET_PATH)
    
    return dataset


def load_dataset(
    connectivity: str = 'organ', batch_size: int = 8,
    filepath: str = '', timestamp: dt.datetime = dt.datetime.today(),
    verbose: int = 0
):
    
    dataset = create_dataset(connectivity=connectivity,
                             filepath=filepath, timestamp=timestamp, verbose=verbose)
        
    idx_train_end = int(len(dataset) * .5)
    idx_valid_end = int(len(dataset) * .75)

    batch_test = len(dataset) - idx_valid_end

    # In the test loader we set the natch size to be
    # equal to the size of the whole test set
    loader_train = DataLoader(dataset[:idx_train_end],
                              batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(dataset[idx_train_end:idx_valid_end],
                              batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset[idx_valid_end:],
                             batch_size=batch_test, shuffle=False)

    return loader_train, loader_valid, loader_test


def classify_response(row):
    # Compare current vs. baseline
    if row.tmtv < row.baseline:
        return 'response'
    elif row.tmtv == row.baseline:
        return 'baseline'
    else:
        return 'progression'


def standardized(t: torch.tensor, mean: torch.tensor, std: torch.tensor) -> torch.tensor:
    """
    Standardize tensor following given mean and standard deviation

    Args:
        t (torch.tensor): tensor to be standardized
        mean (torch.tensor): mean
        std (torch.tensor): standard deviation

    Returns:
        torch.tensor: standardized tensor
    """
    
    return t.sub_(mean).div_(std)


def extract_study_phase(n: str) -> int:
    # split study name into 'pre'/'post' and it's associated number
    status, number = n.split('-')
    # format it with 0 being the treatment start
    return (-1 if status == 'pre' else 1) * int(number)


def format_number_header(heading: str, spotlight: str, footer: str) -> str:
    return f"""
        <div class="container-number">
            <div class="number-header"> {heading} </div>
            <h1 class="number"> {spotlight} </h1>
            <div class="number-footer"> {footer} </div>
        </div>
    """

def preprocess(
    df: pd.DataFrame,
    features_categorical: List[str] = [], features_numerical: List[str] = [],
    training: bool = True, index: str = 'gpcr_id',
    imputer_categorical: BaseEstimator = SimpleImputer(strategy='most_frequent'), 
    imputer_numerical: BaseEstimator = SimpleImputer(strategy='median'),
    estimator_categorical: BaseEstimator = OneHotEncoder(drop='if_binary'),
    estimator_numerical: BaseEstimator = StandardScaler(),
):
    
    processed = df[[index]]
    columns = [index]
    
    if len(features_categorical) > 0:
        if training:
            imputer_categorical.fit(df[features_categorical])
            estimator_categorical.fit(df[features_categorical])
            
        processed = np.append(processed,
                  estimator_categorical.transform(
                      imputer_categorical.transform(df[features_categorical])
                  ).todense(),
                  axis=1)
        
        columns.extend(estimator_categorical.get_feature_names())

    if len(features_numerical) > 0:
        if training:
            imputer_numerical.fit(df[features_numerical])
            estimator_numerical.fit(df[features_numerical])
            
        processed = np.append(processed,
                  estimator_numerical.transform(
                      imputer_numerical.transform(df[features_numerical])),
                  axis=1)
                       
        columns.extend(features_numerical)
    
    processed = pd.DataFrame(processed, columns=columns)
    
    return processed, \
        imputer_categorical, imputer_numerical, estimator_categorical, estimator_numerical