import os

from typing import List, Tuple
from collections.abc import Callable

from itertools import permutations

import pandas as pd
import numpy as np

from scipy.stats import wasserstein_distance

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.transforms import ToDense

BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
DATA_DIR = os.path.join(BASE_DIR, 'data/')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets/')

# CONNECTION_DIR = '/Volumes/lts4-immuno/'
CONNECTION_DIR = '/Users/arnauddhaene/Downloads/'
DATA_FOLDERS = ['data_2021-09-20', 'data_2021-10-04', 'data_2021-10-12']

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
    )
}


def load_dataset(
    connectivity: str = 'wasserstein', batch_size: int = 8,
    test_size: float = 0.2, val_size: float = 0.1, seed: int = 27,
    dense: bool = False, verbose: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get training, validation, and testing DataLoaders.
    Mainly used as a high-level data fetcher in the running script.

    Args:
        connectivity (str, optional): node connectivity method.. Defaults to 'wasserstein'.
        batch_size (int, optional): [description]. Defaults to 8.
        test_size (float, optional): Ratio of test set. Defaults to 0.2.
        val_size (float, optional): Ratio of validation set. Defaults to 0.1.
        seed (int, optional): Random seed. Defaults to 27.
        dense (bool, optional): Output a DenseDataLoader
        verbose (int, optional): tuneable parameter for output verbosity. Defaults to 1.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]:
            * loader_train (DataLoader): packaged training dataset.
            * loader_val (DataLoader): packaged validation dataset.
            * loader_test (DataLoader): packaged testing dataset.
    """

    labels, lesions, patients = fetch_data(verbose)
    
    X_train, X_val, X_test, y_train, y_val, y_test = \
        preprocess(labels, lesions, patients,
                   test_size=test_size, val_size=val_size, seed=seed,
                   verbose=verbose)
        
    loader_train = create_dataset(X=X_train, Y=y_train, dense=dense,
                                  batch_size=batch_size,
                                  connectivity=connectivity, verbose=verbose)
    loader_val = create_dataset(X=X_val, Y=y_val, dense=dense,
                                batch_size=batch_size,
                                connectivity=connectivity, verbose=verbose)
    
    # In the test loader we set the batch size to be
    # equal to the size of the whole test set
    loader_test = create_dataset(X=X_test, Y=y_test, dense=dense,
                                 batch_size=len(y_test), shuffle=False,
                                 connectivity=connectivity, verbose=verbose)
    
    if verbose > 0:
        print('Final amount of datapoints \n' \
              + f'  Train: {len(loader_train.dataset)} \n' \
              + f'  Validation: {len(loader_val.dataset)} \n' \
              + f'  Test: {len(loader_test.dataset)}')

    return loader_train, loader_val, loader_test


def create_dataset(
    X: pd.DataFrame, Y: pd.Series, dense: bool = False, batch_size: int = 8,
    shuffle: bool = True, connectivity: str = 'wasserstein',
    distance: float = 0.5, verbose: int = 0
) -> DataLoader:
    """Packages preprocessed data and its labels into a `torch.utils.data.DataLoader`

    Args:
        X (pd.DataFrame): `gpcr_id` indexed datapoints (lesions)
        Y (pd.Series): `gpcr_id` indexed labels. 1 is NPD.
        dense (bool, optional): Output a DenseDataLoader
        batch_size (int, optional): DataLoader batch size. Defaults to 8.
        shuffle (bool, optional): shuffle graphs in DataLoader. Defaults to True.
        connectivity (str, optional): node connectivity method. Defaults to 'wasserstein'.
        distance (float, optional): if `wasserstein` connectivity is chosen,
            the threshold distance in order to create an edge between nodes. Defaults to 0.5.
        verbose (int, optional): tuneable parameter for output verbosity.. Defaults to 0.

    Raises:
        ValueError: acceptable values for connectivity are: 'fully', 'organ', and 'wasserstein'

    Returns:
        DataLoader: packaged dataset within a DataLoader instance.
    """
    
    lesions = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[2], FILES[DATA_FOLDERS[2]]['lesions']))
    # Filter out benign lesions and non-post-1 studies
    lesions = lesions[(lesions.pars_classification_petct != 'benign') & (lesions.study_name == 'post-01')]
    
    dataset = []
    skipped = 0
    
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
        
        # Skip single-noded graphs
        if num_nodes < 2:
            skipped += 1
            continue

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
            # Use the mean and std of the SUV of each lesion to simulate SUV distributions (normal)
            # and subsequently connect nodes with similar SUV distributions using the Wasserstein distance
            # as a distance metric
            for i in range(num_nodes):
                source_mean, source_sd = pdf.loc[i].mean_suv_val, pdf.loc[i].sd_suv_val
                source_distribution = np.random.normal(source_mean, source_sd, 1000)
                
                targets = [id for id, mu, sd in zip(pdf.index, pdf.mean_suv_val, pdf.sd_suv_val)
                           if wasserstein_distance(source_distribution,
                                                   np.random.normal(mu, sd, 1000)) < distance]

                edge_index.extend([[i, j] for j in targets])
            
        else:
            raise ValueError(f'Connectivity value not accepted: {connectivity}.'
                             "Must be either 'fully', 'wasserstein', or 'organ'.")

        edge_index = torch.tensor(edge_index).t().long()
    
        x = torch.tensor(X.loc[patient].reset_index(drop=True).to_numpy().astype(np.float32))
        y = torch.tensor(Y.loc[patient])

        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, y=y.reshape(-1))

        dataset.append(to_dense(data) if dense else data)
        
    if verbose > 0 and skipped > 0:
        print(f'Skipped {skipped} graphs as they have less than 2 nodes.')
        
    return DenseDataLoader(dataset, batch_size=batch_size, shuffle=shuffle) if dense else \
        DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
        
    def get_feature_names(self) -> List[str]:
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
                If None, will call `get_feature_names`.

        Returns:
            pd.DataFrame: output data.
        """
        if index is None:
            index = df.index
           
        if columns is None:
            columns = self.get_feature_names()
    
        return pd.DataFrame(self.pipe.transform(df), index=index, columns=columns)


def preprocess(
    labels: pd.Series, lesions: pd.DataFrame, patients: pd.DataFrame,
    test_size: float = 0.2, val_size: float = 0.2, seed: int = 27, verbose: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Preprocess filtered raw data into train, validation, and test splits.
    Imputation, standardization, and one-hot encoding of features using sklearn pipelines.

    Args:
        labels (pd.Series): `gpcr_id` indexed Series with progression labels. 1 is NPD.
        lesions (pd.DataFrame): gpcr_id` indexed lesion-level data.
        patients (pd.DataFrame): `gpcr_id` indexed patient-level data including blood screens.
        test_size (float, optional): Ratio of test set. Defaults to 0.2.
        val_size (float, optional): Ratio of validation set. Defaults to 0.1.
        seed (int, optional): Random seed. Defaults to 27.
        verbose (int, optional): tuneable parameter for output verbosity. Defaults to 1.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
            * X_train (pd.DataFrame): `gpcr_id` indexed training dataset.
            * X_val (pd.DataFrame): `gpcr_id` indexed validation dataset.
            * X_test (pd.DataFrame): `gpcr_id` indexed testing dataset.
            * y_train (pd.Series): `gpcr_id` indexed training Series with progression labels. 1 is NPD.
            * y_val (pd.Series): `gpcr_id` indexed validation Series with progression labels. 1 is NPD.
            * y_test (pd.Series): `gpcr_id` indexed test Series with progression labels. 1 is NPD.
    """
    
    I_train, I_test, y_train, y_test = \
        train_test_split(labels.index, labels, test_size=test_size, random_state=seed)
        
    # Account for already taken up test size
    val_size = val_size / (1 - test_size)
    # Compute validation set indices
    I_train, I_val, y_train, y_val = \
        train_test_split(I_train, y_train, test_size=val_size, random_state=seed)
        
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
            ('one-hot', OneHotEncoder(handle_unknown='ignore'),
             features_range[bp[0]:bp[1]]),
            ('count-vec', CountVectorizer(analyzer=set), features_range[bp[1]:bp[2]][0])
        ], remainder='passthrough')),
    ])

    patients_pp = Preprocessor(
        pipe=clf_patients,
        feats_out_fn=lambda c: (c.named_steps['imputers'].transformers_[0][2] \
                                + list(c.named_steps['preprocess'].transformers_[1][1].get_feature_names()) \
                                + c.named_steps['preprocess'].transformers_[2][1].get_feature_names())
    )

    lesions_pp.fit(lesions.loc[I_train])

    lesions_train = lesions_pp.transform(lesions.loc[I_train])
    lesions_val = lesions_pp.transform(lesions.loc[I_val])
    lesions_test = lesions_pp.transform(lesions.loc[I_test])
    
    patients_pp.fit(patients.loc[I_train])

    patients_train = patients_pp.transform(patients.loc[I_train])
    patients_val = patients_pp.transform(patients.loc[I_val])
    patients_test = patients_pp.transform(patients.loc[I_test])
    
    X_train = pd.merge(lesions_train, patients_train, left_index=True, right_index=True)
    X_val = pd.merge(lesions_val, patients_val, left_index=True, right_index=True)
    X_test = pd.merge(lesions_test, patients_test, left_index=True, right_index=True)
    
    if verbose > 0:
        print('Processed and split dataset into \n' \
              + f'  Train: {y_train.shape[0]} \n' \
              + f'  Validation: {y_val.shape[0]} \n' \
              + f'  Test: {y_test.shape[0]}')
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def fetch_data(verbose: int = 1) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
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
    # Keep only radiomics features and assigned organ
    radiomics_features = ['vol_ccm', 'max_suv_val', 'mean_suv_val', 'min_suv_val', 'sd_suv_val']
    lesions = lesions[['gpcr_id', 'study_name', *radiomics_features, 'assigned_organ']]

    if verbose > 0:
        print(f'Post-1 study lesions extracted for {len(lesions.gpcr_id.unique())} patients')

    # LABELS
    progression = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[1],
                                           FILES[DATA_FOLDERS[1]]['progression']))
    progression['pseudorecist'] = progression.pseudorecist.eq('NPD').mul(1)

    # We need to filter out studies who do not have an associated progression label
    # Add prediction score label from progression df
    lesions = lesions.merge(progression[['gpcr_id', 'study_name', 'pseudorecist']],
                            on=['gpcr_id', 'study_name'], how='inner')
    lesions = lesions[lesions.pseudorecist.notna()]
    lesions.drop(columns='pseudorecist', inplace=True)

    if verbose > 0:
        print(f'Post-1 study labels added for {len(lesions.gpcr_id.unique())} patients')
        
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
                      'concomittant_LAG3', 'prior_targeted_therapy', 'prior_treatment',
                      'nivo_maintenance']
    
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
