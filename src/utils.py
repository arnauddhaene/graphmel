import os

from typing import List, Tuple
from collections.abc import Callable

import pickle
import string
import re

import datetime as dt

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator

import torch
from torch_geometric.data import Data

BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
DATA_DIR = os.path.join(BASE_DIR, 'data/')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets/')
CHECKPOINTS_DIR = os.path.join(DATA_DIR, 'checkpoints/')

# CONNECTION_DIR = '/Volumes/lts4-immuno/'
CONNECTION_DIR = '/Users/adhaene/Downloads/'
DATA_FOLDERS = ['data_2021-09-20', 'data_2021-10-04', 'data_2021-10-12', 'data_2021-11-06', 'data_2021-11-18',
                'data_2021-11-26']

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
    ),
    'data_2021-11-18': dict(
        pet='melanoma_lesion-info_radiomics-pet_2021-11-18_anonymized.csv',
        shape='melanoma_lesion-info_radiomics-pet_shape_firstorder_2021-11-18_anonymized.csv',
        lesions='melanoma_lesion-info_organ-overlap_2021-11-18_cleaned_all_anonymized.csv',
        progression='melanoma_petct-exams_progression-status_2021-11-18_anonymized.csv',
        blood='melanoma_info_patient-treatment-blood-mutation_2021-11-18_anonymized.csv'
    ),
    'data_2021-11-26': dict(
        distances='pre-post-01-wasserstein-distances.csv'
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
    distance: float = 0.5, suspicious: float = 0.5, dense: bool = False, verbose: int = 0
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
    
    identifier = (connectivity, round(test_size * 100), seed, dense, \
                  round(distance * 100), round(suspicious * 100))
    
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

        labels, lesions, patients = fetch_data(suspicious=suspicious, verbose=verbose)
        
        X_train, X_test, y_train, y_test = \
            preprocess(labels, lesions, patients,
                       test_size=test_size, seed=seed, verbose=verbose)
            
        dataset_train = create_dataset(X=X_train, Y=y_train, distance=distance,
                                       connectivity=connectivity, verbose=verbose)
        if test_size > 0.:
            dataset_test = create_dataset(X=X_test, Y=y_test, distance=distance,
                                          connectivity=connectivity, verbose=verbose)
        else:
            dataset_test = []
    
        fpath = os.path.join(CHECKPOINTS_DIR,
                             f'{connectivity}_{round(test_size * 100)}_{seed}_{dense}_'
                             f'{round(distance * 100)}_{round(suspicious * 100)}_{dt.date.today()}.pt')
        
        outfile = open(fpath, 'wb')
        pickle.dump((dataset_train, dataset_test), outfile)
        
    if verbose > 0:
        print(f'Final dataset split -> Train: {len(dataset_train)} | Test: {len(dataset_test)}')
    
    return dataset_train, dataset_test


def create_dataset(
    X: pd.DataFrame, Y: pd.Series, connectivity: str = 'wasserstein',
    distance: float = 2.5, verbose: int = 0
) -> List[Data]:
    """Packages preprocessed data and its labels into a dataset

    Args:
        X (pd.DataFrame): `gpcr_id` indexed datapoints (lesions)
        Y (pd.Series): `gpcr_id` indexed labels. 1 is NPD.
        connectivity (str, optional): node connectivity method. Defaults to 'wasserstein'.
        distance (float, optional): if `wasserstein` connectivity is chosen,
            the threshold distance in order to create an edge between nodes. Defaults to 0.5.
        verbose (int, optional): tuneable parameter for output verbosity.. Defaults to 0.

    Raises:
        ValueError: acceptable values for connectivity are: 'fully', 'organ', and 'wasserstein'

    Returns:
        dataset
    """
    
    X.reset_index(inplace=True)
    
    distances = pd.read_csv(
        os.path.join(CONNECTION_DIR, DATA_FOLDERS[5], FILES[DATA_FOLDERS[5]]['distances']))
    distances.study_name = distances.study_name.apply(lambda sn: '-'.join(sn.split('_')))
    
    progression = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[1],
                                           FILES[DATA_FOLDERS[1]]['progression']))
    progression['pseudorecist'] = progression.pseudorecist.eq('NPD').mul(1)
    
    all_labels = progression[progression.study_name.isin(['pre-01', 'post-01', 'post-02'])] \
        .pivot(index='gpcr_id', columns='study_name', values='pseudorecist') \
        .loc[X.gpcr_id]

    # Get rid of invalid distances
    valid_lesions_per_patient = X.groupby(['gpcr_id', 'study_name']).lesion_label_id.unique().to_dict()
    
    def is_valid_distance(row):

        valid_lesions = valid_lesions_per_patient.get((int(row.gpcr_id), row.study_name))
        
        if valid_lesions is None:
            return False

        intersection = set(valid_lesions) & set([row.lesion_i, row.lesion_j])
        
        return len(intersection) == 2

    distances['valid'] = distances.apply(is_valid_distance, axis=1)
    distances = distances[distances.valid]
    
    node_columns = ['original_shape_flatness', 'original_shape_leastaxislength',
                    'original_shape_majoraxislength', 'original_shape_maximum2ddiametercolumn',
                    'original_shape_maximum2ddiameterrow', 'original_shape_maximum2ddiameterslice',
                    'original_shape_maximum3ddiameter', 'original_shape_meshvolume',
                    'original_shape_minoraxislength', 'original_shape_sphericity',
                    'original_shape_surfacearea', 'original_shape_surfacevolumeratio',
                    'original_shape_voxelvolume',
                    'suv_skewness', 'suv_entropy', 'suv_kurtosis', 'suv_uniformity', 'suv_energy',
                    'mtv', 'tlg', 'bones_abdomen',
                    'bones_lowerlimb', 'bones_thorax', 'kidney', 'liver', 'lung',
                    'lymphnode_abdomen', 'lymphnode_lowerlimb', 'lymphnode_thorax',
                    'other_abdomen', 'other_lowerlimb', 'other_thorax', 'spleen']
    
    # node_columns = ['original_shape_sphericity', 'original_shape_surfacevolumeratio',
    #                 'suv_skewness', 'suv_kurtosis', 'mtv', 'tlg', 'bones_abdomen',
    #                 'bones_lowerlimb', 'bones_thorax', 'kidney', 'liver', 'lung',
    #                 'lymphnode_abdomen', 'lymphnode_lowerlimb', 'lymphnode_thorax',
    #                 'other_abdomen', 'other_lowerlimb', 'other_thorax', 'spleen']
    patient_columns = ['age_at_treatment_start_in_years', 'x0_male']
    
    study_columns = [str(feat) for feat in list(X.columns) \
                     if (feat not in node_columns \
                         and feat not in patient_columns \
                         and feat not in ['gpcr_id', 'study_name', 'lesion_label_id'])]

    dataset = []

    for patient in list(X.gpcr_id.unique()):

        pdf = X[X.gpcr_id == patient]

        edge_index, edge_weight, split_sizes, graph_sizes, order = \
            np.array([]).reshape(0, 2), np.array([]), [], [], []

        # Iterate over studies
        studies_ordered = sorted(map(lambda s: (s, extract_study_phase(s)), pdf.study_name.unique()),
                                 key=lambda kv: kv[1])
        for study, _ in studies_ordered:

            study_edge_index = []
            
            psdf = pdf[pdf.study_name == study].rename_axis('order').reset_index()
            
            # Fetch order to make sure final extracted features are attributed to correct node
            order.extend(psdf['order'])

            num_nodes = psdf.shape[0]
            graph_sizes.append(num_nodes)

            wanted_edges = distances[(distances.gpcr_id == patient) & (distances.study_name == study) \
                                     & (distances.wasserstein_distance < distance)]

            study_edge_index = wanted_edges[['lesion_i', 'lesion_j']].to_numpy().astype(int)
            
            # Replace lesion_label_id by index from preprocessed data
            # Inspired from Method #3 of:
            # https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python
            keys, values = psdf.lesion_label_id, psdf.index
            mapping = np.zeros(keys.max() + 1, dtype=values.dtype)
            mapping[keys] = values
            study_edge_index = mapping[study_edge_index]
            
            split_sizes.append(study_edge_index.shape[0] * 2)

            # Add edge weight
            study_edge_weight = wanted_edges.wasserstein_distance.to_numpy().astype(float)
            edge_weight = np.concatenate([edge_weight, study_edge_weight, study_edge_weight])

            # Add edges in both directions
            edge_index = np.concatenate([edge_index, study_edge_index, np.flip(study_edge_index, axis=1)])

            # Skip graph if there are no edges
            if edge_index.shape[1] == 0:
                continue

        edge_index = torch.tensor(edge_index).t().long()
        edge_weight = torch.tensor(edge_weight).float()
        # Can be used with {edge_index, edge_weight}.split(tuple(split_sizes)) to separate edges
        split_sizes = torch.tensor(split_sizes).long()
        # Can be used with x.split(tuple(graph_sizes)) to separate node features
        graph_sizes = torch.tensor(graph_sizes).long()
        
        x = torch.tensor(pdf.loc[order][node_columns].to_numpy().astype(np.float32))
        # Use unique to fetch only one row per study
        study_features = torch.tensor(pdf.loc[order][study_columns].to_numpy()).unique(dim=0)
        
        if study_features.shape[0] < graph_sizes.shape[0]:
            study_features = study_features.repeat(graph_sizes.shape[0], 1)
        
        # Select only the first row as they are identical for both studies
        patient_features = torch.tensor(pdf.loc[order][patient_columns].to_numpy()[0, :])

        y = torch.tensor(Y.loc[patient])
        aux_y = torch.tensor(all_labels.loc[patient][dict(studies_ordered).keys()].to_numpy())

        data = Data(x=x, study_features=study_features, patient_features=patient_features,
                    edge_index=edge_index, graph_sizes=graph_sizes, split_sizes=split_sizes,
                    y=y.reshape(-1), aux_y=aux_y, edge_weight=edge_weight, num_nodes=x.shape[0])
        
        dataset.append(data)
        
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
        
    lesions_pp = Preprocessor(
        pipe=ColumnTransformer(
            [('scaler', StandardScaler(), make_column_selector(dtype_include=np.number)),
             ('one-hot', OneHotEncoder(), make_column_selector(dtype_include=object))]),
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
    
    if test_size == 0.:
        I_train, y_train = labels.index, labels
        I_test, y_test = [], None
    else:
        I_train, I_test, y_train, y_test = \
            train_test_split(labels.index, labels, test_size=test_size, random_state=seed)

    lesions_pp.fit(lesions.loc[I_train])
    lesions_train = lesions_pp.transform(lesions.loc[I_train])

    patients_pp.fit(patients.loc[I_train])
    patients_train = patients_pp.transform(patients.loc[I_train])

    X_train = pd.merge(lesions_train, patients_train, left_index=True, right_index=True)
    
    if test_size > 0.:
        lesions_test = lesions_pp.transform(lesions.loc[I_test])
        patients_test = patients_pp.transform(patients.loc[I_test])
    
        X_test = pd.merge(lesions_test, patients_test, left_index=True, right_index=True)
    else:
        X_test = None
    
    # TODO: somehow avoid the merging of these two as the intermediate step means
    # we need to store more data (duplicating blood data over all lesions)
    # --> this would imply changing the way we create the dataset
    #     should totally be possible however
    
    return X_train, X_test, y_train, y_test


def fetch_data(suspicious: float, verbose: int = 0) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Fetch data from sources explicited in __file__ constants. First filtering of raw data.

    Args:
        suspicious (float): threshold of malignancy supsicion for addition into dataset
        verbose (int, optional): tuneable parameter for output verbosity. Defaults to 1.

    Returns:
        Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
            * labels (pd.Series): `gpcr_id` indexed Series with progression labels. 1 is NPD.
            * lesions (pd.DataFrame): `gpcr_id` indexed lesion-level data.
            * patients (pd.DataFrame): `gpcr_id` indexed patient-level data including blood screens.
            
    """
    # LESIONS
    # Fetch
    lesions = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[4], FILES[DATA_FOLDERS[4]]['lesions']))
    shape = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[4], FILES[DATA_FOLDERS[4]]['shape']),
                        index_col=0)
    shape = standardise_column_names(shape)

    # Merge with radiomics
    lesions = lesions.merge(shape, on=['gpcr_id', 'study_name', 'roi_id'], how='inner')

    # Filter out benign lesions and non-post-1 studies
    lesions = lesions[(lesions.pars_suspicious_prob_petct > suspicious) \
                      & (lesions.study_name.isin(['pre-01', 'post-01']))]

    # Filter out single-lesion studies
    multiple_lesions = lesions.groupby(['gpcr_id', 'study_name']).size().gt(1)
    multiple_lesions = multiple_lesions.index[multiple_lesions.values]
    lesions = lesions.set_index(['gpcr_id', 'study_name']).loc[multiple_lesions].reset_index()

    # Keep only radiomics features and assigned organ
    # radiomics_features = ['original_shape_sphericity', 'original_shape_surfacevolumeratio', 'mtv', 'tlg',
    #                       'suv_skewness', 'suv_kurtosis', 'pars_suspicious_prob_petct']
    radiomics_features = [
        'original_shape_elongation', 'original_shape_flatness', 'original_shape_leastaxislength',
        'original_shape_majoraxislength', 'original_shape_maximum2ddiametercolumn',
        'original_shape_maximum2ddiameterrow', 'original_shape_maximum2ddiameterslice',
        'original_shape_maximum3ddiameter', 'original_shape_meshvolume', 'original_shape_minoraxislength',
        'original_shape_sphericity', 'original_shape_surfacearea', 'original_shape_surfacevolumeratio',
        'original_shape_voxelvolume', 'mtv', 'tlg', 'pars_suspicious_prob_petct',
        'suv_skewness', 'suv_entropy', 'suv_kurtosis', 'suv_uniformity', 'suv_energy']

    lesions = lesions[['gpcr_id', 'study_name', 'lesion_label_id', *radiomics_features, 'assigned_organ']]

    if verbose > 0:
        print(f"Post-1 study lesions extracted for {len(lesions.gpcr_id.unique())} patients")
    
    # LABELS
    progression = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[1],
                                           FILES[DATA_FOLDERS[1]]['progression']))
    progression['pseudorecist'] = progression.pseudorecist.eq('NPD').mul(1)

    # We need to filter out studies who do not have an associated progression label
    # Add label from progression DataFrame
    lesions = lesions.merge(progression[progression.study_name == 'post-02'][['gpcr_id', 'pseudorecist']],
                            on=['gpcr_id'], how='inner')
    lesions = lesions[lesions.pseudorecist.notna()]
    lesions.drop(columns='pseudorecist', inplace=True)

    if verbose > 0:
        print(f"Post-2 study labels added for {len(lesions.gpcr_id.unique())} patients")
        
    # PATIENT-LEVEL
    patients = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[2], FILES[DATA_FOLDERS[2]]['patients']))
    studies = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[0], FILES[DATA_FOLDERS[0]]['studies']))
    # Fix encoding for 90+ patients
    patients['age_at_treatment_start_in_years'] = \
        patients.age_at_treatment_start_in_years.apply(lambda a: 90 if a == '90 or older' else int(a))

    blood = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[4], FILES[DATA_FOLDERS[4]]['blood']))
    blood.rename(columns={feature: feature.replace('-', '_') for feature in blood.columns}, inplace=True)
    # Listify immunotherapy type to create multi-feature encoding
    blood['immuno_therapy_type'] = blood.immuno_therapy_type \
        .apply(lambda t: ['ipi', 'nivo'] if t == 'ipinivo' else [t])

    # Filter in the patient information that we want access to
    patient_features = ['age_at_treatment_start_in_years']
    blood_features = ['sex', 'bmi', 'performance_score_ecog', 'ldh_sang_ul', 'neutro_absolus_gl',
                      'eosini_absolus_gl', 'leucocytes_sang_gl', 'NRAS_MUTATION', 'BRAF_MUTATION',
                      'immuno_therapy_type',
                      ##
                      'prior_targeted_therapy', 'prior_treatment', 'nivo_maintenance', 'lympho_absolus_gl',
                      'concomittant_tvec']

    patients = patients[['gpcr_id', *patient_features]]
    blood = blood[['gpcr_id', 'n_days_to_treatment_start', *blood_features]]
    blood['study_name'] = 'blood'

    radiomics = pd.read_csv(os.path.join(CONNECTION_DIR, DATA_FOLDERS[3],
                                         FILES[DATA_FOLDERS[3]]['radiomics']))

    potential_patients = list(set(lesions[lesions.study_name == 'post-01'].gpcr_id) \
                              & set(patients.gpcr_id) & set(radiomics.gpcr_id) & set(blood.gpcr_id) \
                              & set(progression[progression.study_name == 'post-02'].gpcr_id) \
                              & set(studies[studies.study_name.isin(['pre-01', 'post-01'])].gpcr_id))
 
    merged_studies = pd.concat([blood, studies[['gpcr_id', 'study_name', 'n_days_to_treatment_start']]])
    
    blood_processed = pd.DataFrame()

    for patient in potential_patients:
            
        patient_studies = merged_studies[merged_studies.gpcr_id == patient] \
            .set_index('n_days_to_treatment_start').sort_index()
        
        # Linear interpolation for numeric values
        # Follow by backwards fill, then forward fill for the rest
        filled_studies = patient_studies.interpolate(method='index').bfill().ffill().reset_index()
            
        blood_processed = pd.concat([
            blood_processed, filled_studies[filled_studies.study_name.isin(['pre-01', 'post-01'])]])

    # Transform all one-hot encoded features into True/False to avoid scaler
    for feature in blood_features:
        values = blood_processed[feature].value_counts().keys()
        if len(values) == 2 and all(values == [0, 1]):
            blood_processed[feature] = blood_processed[feature].astype(bool)
        
    blood_processed.reset_index(inplace=True, drop=True)

    progression.set_index('gpcr_id', inplace=True)
    labels = progression[progression.study_name == 'post-02'].loc[potential_patients].pseudorecist

    if verbose > 0:
        print(f'The intersection of datasets showed {len(potential_patients)} potential datapoints.')

    # Prepare for return
    lesions = lesions[lesions.gpcr_id.isin(potential_patients)]
    lesions.set_index(['gpcr_id', 'study_name', 'lesion_label_id'], inplace=True)
    patients = blood_processed.merge(patients, on='gpcr_id', how='inner')
    patients.set_index(['gpcr_id', 'study_name'], inplace=True)

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


def standardise_column_names(df, remove_punct=True):
    """ Converts all DataFrame column names to lower case replacing
    whitespace of any length with a single underscore. Can also strip
    all punctuation from column names.
    
    Taken from: https://gist.github.com/georgerichardson/db66b686b4369de9e7196a65df45fc37
    
    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame with non-standardised column names.
    remove_punct: bool (default True)
        If True will remove all punctuation from column names.
    
    Returns
    -------
    df: pandas.DataFrame
        DataFrame with standardised column names.
    Example
    -------
    >>> df = pd.DataFrame({'Column With Spaces': [1,2,3,4,5],
                           'Column-With-Hyphens&Others/': [6,7,8,9,10],
                           'Too    Many Spaces': [11,12,13,14,15],
                           })
    >>> df = standardise_column_names(df)
    >>> print(df.columns)
    Index(['column_with_spaces',
           'column_with_hyphens_others',
           'too_many_spaces'], dtype='object')
    """
    
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    for c in df.columns:
        c_mod = c.lower()
        if remove_punct:
            c_mod = c_mod.translate(translator)
        c_mod = '_'.join(c_mod.split(' '))
        if c_mod[-1] == '_':
            c_mod = c_mod[:-1]
        c_mod = re.sub(r'\_+', '_', c_mod)
        df.rename({c: c_mod}, inplace=True, axis=1)
    return df


def correlation_matrix(df: pd.DataFrame):
    """
    A function to calculate and plot
    correlation matrix of a DataFrame.
    """
    # Create the matrix
    matrix = df.corr()
    
    # Create cmap
    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                                 n=9, center="light", as_cmap=True)
    # Create a mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    
    # Make figsize bigger
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot the matrix
    _ = sns.heatmap(matrix, mask=mask, center=0, annot=True,
                    fmt='.2f', square=True, cmap=cmap, ax=ax)
