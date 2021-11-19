import os

from typing import List, Tuple
from collections.abc import Callable

import pickle

import datetime as dt

from itertools import permutations

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToDense

BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
DATA_DIR = os.path.join(BASE_DIR, 'data/')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets/')
CHECKPOINTS_DIR = os.path.join(DATA_DIR, 'checkpoints/')

# CONNECTION_DIR = '/Volumes/lts4-immuno/'
CONNECTION_DIR = '/Users/adhaene/Downloads/'
DATA_FOLDERS = ['data_2021-09-20', 'data_2021-10-04', 'data_2021-10-12', 'data_2021-11-06']

FILES = {
    'data_2021-09-20': dict(
        lesions='melanoma_lesion-info_organ-overlap_2021-09-17_anonymized_cleaned_all.csv',
        lesion_mapping='melanoma_lesion_mapping_2021-09-20_anonymized.csv',
        patients='melanoma_patient-level_summary_anonymized.csv',
        studies='melanoma_study_level_summary_anonymized.csv'
    ),
    'data_2021-10-04': dict(
        blood='melanoma_info_patient-treatment-blood-mutation_2021-10-04_anonymized.csv',
        progression='melanoma_petct-exams_progression-status_2021-10-04_anonymized.csv'
    ),
    'data_2021-10-12': dict(
        lesions='melanoma_lesion-info_organ-overlap_2021-10-12_anonymized_cleaned_all.csv',
        patients='melanoma_patient-level_summary_anonymized.csv',
        studies='melanoma_study_level_summary_anonymized.csv'
    ),
    'data_2021-11-06': dict(
        radiomics='melanoma_lesions-radiomics.csv',
        distances='post-01-wasserstein-distances.csv'
    )
}


def fetch_dataset_id(fname: str) -> Tuple[str, int, int, bool]:
    
    name, extension = tuple(fname.split('.'))
    
    connectivity, test_size, seed, dense = tuple(name.split('_')[:4])
    
    test_size_p = int(test_size)
    seed = int(seed)
    dense = dense == 'True'
    
    return (connectivity, test_size_p, seed, dense)


def load_dataset(
    connectivity: str = 'wasserstein', test_size: float = 0.2, seed: int = 27,
    distance: float = 0.5, dense: bool = True, verbose: int = 0
) -> Tuple[List[Data], List[Data]]:
    """
    Get training, validation, and testing DataLoaders.
    Mainly used as a high-level data fetcher in the running script.

    Args:
        connectivity (str, optional): node connectivity method.. Defaults to 'wasserstein'.
        test_size (float, optional): Ratio of test set. Defaults to 0.2.
        distance (float, optional): Wasserstein distance threshold for graph creation. Defaults to 0.5.
        seed (int, optional): Random seed. Defaults to 27.
        dense (bool, optional): Output a DenseDataLoader
        verbose (int, optional): tuneable parameter for output verbosity. Defaults to 1.

    Returns:
        Tuple[List[Data], List[Data]]:
            * loader_train (List[Data]): packaged training dataset.
            * loader_test (List[Data]): packaged testing dataset.
    """
    
    identifier = (connectivity, round(test_size * 100), seed, dense)
    
    files = os.listdir(CHECKPOINTS_DIR)
    stored_datasets = dict(zip(map(fetch_dataset_id, files), files))
    
    if identifier in stored_datasets.keys():
        
        fpath = os.path.join(CHECKPOINTS_DIR, stored_datasets[identifier])
        
        if verbose > 0:
            date = stored_datasets[identifier].split('.')[0].split('_')[-1]
            print(f'Using stored dataset that was saved on {date}.')
        
        infile = open(fpath, 'rb')
        dataset_train, dataset_test = pickle.load(infile)
        
    else:

        labels, lesions, patients = fetch_data(verbose)
        
        X_train, X_test, y_train, y_test = \
            preprocess(labels, lesions, patients,
                       test_size=test_size, seed=seed, verbose=verbose)
            
        dataset_train = create_dataset(X=X_train, Y=y_train, dense=dense, distance=distance,
                                       connectivity=connectivity, verbose=verbose)
        
        # In the test loader we set the batch size to be
        # equal to the size of the whole test set
        dataset_test = create_dataset(X=X_test, Y=y_test, dense=dense, distance=distance,
                                      connectivity=connectivity, verbose=verbose)
    
        fpath = os.path.join(CHECKPOINTS_DIR,
                             f'{connectivity}_{round(test_size * 100)}_{seed}_{dense}_{dt.date.today()}.pt')
        
        outfile = open(fpath, 'wb')
        pickle.dump((dataset_train, dataset_test), outfile)
        
    if verbose > 0:
        print(f'Final dataset split -> Train: {len(dataset_train)} | Test: {len(dataset_test)}')
    
    return dataset_train, dataset_test


def create_dataset(
    X: pd.DataFrame, Y: pd.Series, dense: bool = False, connectivity: str = 'wasserstein',
    distance: float = 0.5, verbose: int = 0
) -> List[Data]:
    """Packages preprocessed data and its labels into a dataset

    Args:
        X (pd.DataFrame): `gpcr_id` indexed datapoints (lesions)
        Y (pd.Series): `gpcr_id` indexed labels. 1 is NPD.
        dense (bool, optional): create dense Graph representations
        connectivity (str, optional): node connectivity method. Defaults to 'wasserstein'.
        distance (float, optional): if `wasserstein` connectivity is chosen,
            the threshold distance in order to create an edge between nodes. Defaults to 0.5.
        verbose (int, optional): tuneable parameter for output verbosity.. Defaults to 0.

    Raises:
        ValueError: acceptable values for connectivity are: 'fully', 'organ', and 'wasserstein'

    Returns:
        dataset
    """
    
    lesions = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[2], FILES[DATA_FOLDERS[2]]['lesions']))
    # Filter out benign lesions and non-post-1 studies
    lesions = lesions[(lesions.pars_classification_petct != 'benign') & (lesions.study_name == 'post-01')]
    radiomics = pd.read_csv(os.path.join(CONNECTION_DIR, DATA_FOLDERS[3],
                                         FILES[DATA_FOLDERS[3]]['radiomics']))
    radiomics['study_name'] = radiomics.study_name.apply(lambda sn: '-'.join(sn[1:].split('_')))
    radiomics_features = ['vol_ccm', 'max_suv_val', 'mean_suv_val', 'min_suv_val', 'sd_suv_val']
    lesions.drop(columns=radiomics_features, inplace=True)
    lesions = lesions.merge(radiomics, on=['gpcr_id', 'study_name', 'lesion_label_id'], how='inner')
    
    distances = pd.read_csv(os.path.join(CONNECTION_DIR, DATA_FOLDERS[3],
                                         FILES[DATA_FOLDERS[3]]['distances']))
    
    # Get rid of invalid distances
    valid_lesions_per_patient = lesions.groupby('gpcr_id').lesion_label_id.unique().to_dict()
    
    def is_valid_distance(row):
    
        valid_lesions = valid_lesions_per_patient.get(int(row.gpcr_id))
        
        if valid_lesions is None:
            return False

        intersection = set(valid_lesions) & set([row.lesion_i, row.lesion_j])
        
        return len(intersection) == 2
    
    distances['valid'] = distances.apply(is_valid_distance, axis=1)
    distances = distances[distances.valid]
    
    dataset = []
    
    if dense:
        max_num_nodes = lesions.groupby('gpcr_id').size().max()
        to_dense = ToDense(max_num_nodes)

    for patient in list(X.index.unique()):

        # Create patient sub-DataFrame of all his post-1 study lesions
        pdf = lesions[lesions.gpcr_id == patient].reset_index()
        
        # Sanity check
        assert pdf.shape[0] == X[X.index == patient].shape[0], f'Unequal lesion count for patient {patient}'
        
        num_nodes = pdf.shape[0]
        edge_index = []
        
        # Connect lesions using different methodologies
        if connectivity == 'organ':
            # Connect all lesions that are assigned to the same organ
            for i in range(num_nodes):
                source = pdf.loc[i].assigned_organ
                targets = list(pdf[pdf.assigned_organ == source].index)

                edge_index.extend([[i, j] for j in targets if i != j])
                
        elif connectivity == 'fully':
            # Create a fully-connected network representation
            edge_index = list(permutations(range(num_nodes), 2, ))
            
        elif connectivity == 'wasserstein':
            wanted_edges = distances[(distances.gpcr_id == patient) \
                                     & (distances.wasserstein_distance < distance)]
            edge_index = wanted_edges[['lesion_i', 'lesion_j']].to_numpy().astype(int)
            
            # Replace lesion_label_id by index from preprocessed data
            # Inspired from Method #3 of:
            # https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python
            keys, values = pdf.lesion_label_id, pdf.index
            mapping = np.zeros(keys.max() + 1, dtype=values.dtype)
            mapping[keys] = values
            edge_index = mapping[edge_index]
            
            # Add edges in both directions
            edge_index = np.concatenate([edge_index, np.flip(edge_index, axis=1)])
            
            # Add edge weight
            edge_weight = wanted_edges.wasserstein_distance.to_numpy().astype(float)
            edge_weight = np.concatenate([edge_weight, edge_weight])
                
        else:
            raise ValueError(f'Connectivity value not accepted: {connectivity}.'
                             "Must be either 'fully', 'wasserstein', or 'organ'.")

        edge_index = torch.tensor(edge_index).t().long()
    
        # Skip graph if there are no edges
        if edge_index.shape[1] == 0:
            continue
        
        all_x = X.loc[patient].reset_index(drop=True).to_numpy().astype(np.float32)
        
        x = torch.tensor(all_x[:, :18])
        y = torch.tensor(Y.loc[patient])
        graph_features = torch.tensor(all_x[0, 18:])
        edge_weight = torch.tensor(edge_weight).float()
        
        extra_kwargs = dict() if dense else dict(edge_weight=edge_weight)
        
        data = Data(x=x, graph_features=graph_features, edge_index=edge_index,
                    num_nodes=num_nodes, y=y.reshape(-1), **extra_kwargs)

        dataset.append(to_dense(data) if dense else data)
        
    return dataset


class Preprocessor:
    """Preprocessor class that acts as a BaseEstimator wrapper for preprocessing pipelining"""
    
    def __init__(self, pipe: BaseEstimator, feats_out_fn: Callable) -> None:
        """Constructor

        Args:
            pipe (BaseEstimator): sklearn pipeline that should be fitted by training data and
                should transform validation and testing data.
            feature_callback (Callable[[BaseEstimator], List[str]]): function that yields list
                feature names available on pipe output.
        """
        self.pipe = pipe
        self.feats_out_fn = feats_out_fn
        
    def get_feature_names_out(self) -> List[str]:
        """Yields list of feature names available on pipe output.

        Returns:
            List[str]: list of feature names available on pipe output.
        """
        return self.feats_out_fn(self.pipe)
    
    def fit(self, df: pd.DataFrame) -> None:
        """Fitting method. To be used with training data `X_train`

        Args:
            df (pd.DataFrame): training data.
        """
        self.pipe.fit(df)
    
    def transform(self, df: pd.DataFrame, index: List = None, columns: List[str] = None) -> pd.DataFrame:
        """Transformer method. To be used on training, validation, and testing data.

        Args:
            df (pd.DataFrame): input data.
            index (List, optional): output `gpcr_id` index. Defaults to None. If None, will use `df.index`.
            columns (List[str], optional): [description]. Defaults to None.
                If None, will call `get_feature_names_out`.

        Returns:
            pd.DataFrame: output data.
        """
        if index is None:
            index = df.index
           
        if columns is None:
            columns = self.get_feature_names_out()
    
        return pd.DataFrame(self.pipe.transform(df), index=index, columns=columns)


def preprocess(
    labels: pd.Series, lesions: pd.DataFrame, patients: pd.DataFrame,
    test_size: float = 0.2, seed: int = 27, verbose: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocess filtered raw data into train, validation, and test splits.
    Imputation, standardization, and one-hot encoding of features using sklearn pipelines.

    Args:
        labels (pd.Series): `gpcr_id` indexed Series with progression labels. 1 is NPD.
        lesions (pd.DataFrame): gpcr_id` indexed lesion-level data.
        patients (pd.DataFrame): `gpcr_id` indexed patient-level data including blood screens.
        test_size (float, optional): Ratio of test set. Defaults to 0.2.
        seed (int, optional): Random seed. Defaults to 27.
        verbose (int, optional): tuneable parameter for output verbosity. Defaults to 1.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
            * X_train (pd.DataFrame): `gpcr_id` indexed training dataset.
            * X_test (pd.DataFrame): `gpcr_id` indexed testing dataset.
            * y_train (pd.Series): `gpcr_id` indexed training Series with progression labels. 1 is NPD.
            * y_test (pd.Series): `gpcr_id` indexed test Series with progression labels. 1 is NPD.
    """
    
    I_train, I_test, y_train, y_test = \
        train_test_split(labels.index, labels, test_size=test_size, random_state=seed)
        
    lesions_pp = Preprocessor(
        pipe=ColumnTransformer(
            [('scaler', StandardScaler(), make_column_selector(dtype_include=np.number)),
             ('one-hot', OneHotEncoder(),
              make_column_selector(dtype_include=object))]),
        feats_out_fn=lambda c: c.transformers_[0][-1] + list(c.transformers_[1][1].categories_[0])
    )
    
    patients_numerical = list(patients.select_dtypes(np.number).columns)
    patients_categorical = list(patients.select_dtypes([bool, object]).columns)
    patients_categorical.remove('immuno_therapy_type')

    features_range = list(range(len(patients_numerical) + len(patients_categorical) + 1))
    bp = np.cumsum([len(patients_numerical), len(patients_categorical), 1])

    clf_patients = Pipeline([
        ('imputers', ColumnTransformer([
            ('median', SimpleImputer(strategy='median'), patients_numerical),
            ('frequent', SimpleImputer(strategy='most_frequent'), patients_categorical)
        ], remainder='passthrough')),
        ('preprocess', ColumnTransformer([
            ('scaler', StandardScaler(), features_range[0:bp[0]]),
            ('one-hot', OneHotEncoder(drop='if_binary', handle_unknown='ignore'),
             features_range[bp[0]:bp[1]]),
            ('count-vec', CountVectorizer(analyzer=set), features_range[bp[1]:bp[2]][0])
        ], remainder='passthrough')),
    ])

    patients_pp = Preprocessor(
        pipe=clf_patients,
        feats_out_fn=lambda c: (c.named_steps['imputers'].transformers_[0][2] \
                                + list(c.named_steps['preprocess'].transformers_[1][1] \
                                .get_feature_names_out()) \
                                + list(c.named_steps['preprocess'].transformers_[2][1] \
                                .get_feature_names_out()))
    )

    lesions_pp.fit(lesions.loc[I_train])

    lesions_train = lesions_pp.transform(lesions.loc[I_train])
    lesions_test = lesions_pp.transform(lesions.loc[I_test])
    
    patients_pp.fit(patients.loc[I_train])

    patients_train = patients_pp.transform(patients.loc[I_train])
    patients_test = patients_pp.transform(patients.loc[I_test])
    
    X_train = pd.merge(lesions_train, patients_train, left_index=True, right_index=True)
    X_test = pd.merge(lesions_test, patients_test, left_index=True, right_index=True)
    
    return X_train, X_test, y_train, y_test


def fetch_data(verbose: int = 0) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Fetch data from sources explicited in __file__ constants. First filtering of raw data.

    Args:
        verbose (int, optional): tuneable parameter for output verbosity. Defaults to 1.

    Returns:
        Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
            * labels (pd.Series): `gpcr_id` indexed Series with progression labels. 1 is NPD.
            * lesions (pd.DataFrame): `gpcr_id` indexed lesion-level data.
            * patients (pd.DataFrame): `gpcr_id` indexed patient-level data including blood screens.
            
    """
    # LESIONS
    lesions = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[2], FILES[DATA_FOLDERS[2]]['lesions']))
    # Filter out benign lesions and non-post-1 studies
    lesions = lesions[(lesions.pars_classification_petct != 'benign') & (lesions.study_name == 'post-01')]
    # Filter out single-lesion studies
    multiple_lesions = lesions.groupby('gpcr_id').size().gt(1)
    multiple_lesions = multiple_lesions.index[multiple_lesions.values]
    lesions = lesions[lesions.gpcr_id.isin(multiple_lesions)]
    # RADIOMICS
    radiomics = pd.read_csv(os.path.join(CONNECTION_DIR, DATA_FOLDERS[3],
                                         FILES[DATA_FOLDERS[3]]['radiomics']))
    radiomics['study_name'] = radiomics.study_name.apply(lambda sn: '-'.join(sn[1:].split('_')))
    radiomics_features = ['vol_ccm', 'max_suv_val', 'mean_suv_val', 'min_suv_val', 'sd_suv_val']
    lesions.drop(columns=radiomics_features, inplace=True)
    lesions = lesions.merge(radiomics, on=['gpcr_id', 'study_name', 'lesion_label_id'], how='inner')
    # Remove volume and add voxels
    # (as they are highly correlated and voxels is more consistent with new values)
    radiomics_features.append('voxels')
    radiomics_features.remove('vol_ccm')
    # Keep only radiomics features and assigned organ
    lesions = lesions[['gpcr_id', 'study_name', *radiomics_features, 'assigned_organ']]

    if verbose > 0:
        print(f"Post-1 study lesions extracted for {len(lesions.gpcr_id.unique())} patients")

    # LABELS
    progression = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[1],
                                           FILES[DATA_FOLDERS[1]]['progression']))
    progression['pseudorecist'] = progression.pseudorecist.eq('NPD').mul(1)

    # We need to filter out studies who do not have an associated progression label
    # Add label from progression DataFrame
    lesions = lesions.merge(progression[['gpcr_id', 'study_name', 'pseudorecist']],
                            on=['gpcr_id', 'study_name'], how='inner')
    lesions = lesions[lesions.pseudorecist.notna()]
    lesions.drop(columns='pseudorecist', inplace=True)

    if verbose > 0:
        print(f"Post-1 study labels added for {len(lesions.gpcr_id.unique())} patients")
        
    # PATIENT-LEVEL
    patients = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[2], FILES[DATA_FOLDERS[2]]['patients']))
    # Fix encoding for 90+ patients
    patients['age_at_treatment_start_in_years'] = \
        patients.age_at_treatment_start_in_years.apply(lambda a: 90 if a == '90 or older' else int(a))

    blood = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[1], FILES[DATA_FOLDERS[1]]['blood']))
    blood.rename(columns={feature: feature.replace('-', '_') for feature in blood.columns}, inplace=True)
    # Listify immunotherapy type to create multi-feature encoding
    blood['immuno_therapy_type'] = blood.immuno_therapy_type \
        .apply(lambda t: ['ipi', 'nivo'] if t == 'ipinivo' else [t])

    # Filter in the patient information that we want access to
    patient_features = ['age_at_treatment_start_in_years']
    blood_features = ['sex', 'bmi', 'performance_score_ecog', 'ldh_sang_ul', 'neutro_absolus_gl',
                      'eosini_absolus_gl', 'leucocytes_sang_gl', 'NRAS_MUTATION', 'BRAF_MUTATION',
                      'immuno_therapy_type', 'lympho_absolus_gl', 'concomittant_tvec',
                      'prior_targeted_therapy', 'prior_treatment', 'nivo_maintenance']
    
    # Transform all one-hot encoded features into True/False to avoid scaler
    for feature in blood_features:
        values = blood[feature].value_counts().keys()
        if len(values) == 2 and all(values == [0, 1]):
            blood[feature] = blood[feature].astype(bool)

    patients = patients[['gpcr_id', *patient_features]]
    blood = blood[['gpcr_id', *blood_features]]
    
    potential_patients = list(set(lesions.gpcr_id) & set(patients.gpcr_id))
    progression.set_index('gpcr_id', inplace=True)
    labels = progression[progression.study_name == 'post-01'].loc[potential_patients].pseudorecist

    if verbose > 0:
        print(f'The intersection of datasets showed {len(potential_patients)} potential datapoints.')
    
    # Prepare for return
    lesions.drop(columns='study_name', inplace=True)
    lesions.set_index('gpcr_id', inplace=True)
    patients = patients.merge(blood, on='gpcr_id', how='inner')
    patients.set_index('gpcr_id', inplace=True)

    return labels, lesions, patients


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
