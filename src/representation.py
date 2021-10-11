import os

from itertools import permutations

import pandas as pd
import numpy as np
import networkx as nx

from scipy.stats import wasserstein_distance

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import streamlit as st

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

from utils import DATA_FOLDERS, FILES, ASSETS_DIR, extract_study_phase, preprocess


pio.templates.default = 'seaborn'
top_h_legend = dict(orientation='h', yanchor="bottom", y=1.02)
figure_labels = {
    'assigned_organ': 'Assigned organ',
    'vol_ccm': 'Volume (ccm)',
    **{f'{metric}_suv_val': f'{metric.capitalize()} SUV' for metric in ['max', 'mean', 'min', 'sd']}
}
WIDTH = 900

st.set_page_config(
    layout="centered",
    page_icon="https://pbs.twimg.com/profile_images/1108310620590559232/7fTy3YtS_400x400.png")

# Styles
with open(os.path.join(ASSETS_DIR, "style.css")) as file:
    styles = file.read()
st.write(f"<style> {styles} </style>", unsafe_allow_html=True)

st.title("Graph analytics for immunotherapy response prediction in melanoma")

CONNECTION_DIR = '/Users/arnauddhaene/Downloads/'

# Fetch data
lesions = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[0], FILES['lesions']))
lesions['study_phase'] = lesions.study_name.apply(extract_study_phase)
lesions = lesions[(lesions.pars_classification_petct != 'benign')]

progression = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[1], FILES['progression']))

studies = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[0], FILES['studies']))
studies.rename(columns={'is_malignant': 'malignant_lesions'}, inplace=True)
studies.drop(columns=['n_days_to_treatment_start', 'n_days_to_treatment_end'], inplace=True)

patients = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[0], FILES['patients']))
patients['age_at_treatment_start_in_years'] = \
    patients.age_at_treatment_start_in_years.apply(lambda a: 90 if a == '90 or older' else int(a))

blood = pd.read_csv(os.path.join(CONNECTION_DIR + DATA_FOLDERS[1], FILES['blood']))
blood.drop(columns=['n_days_to_treatment_start', 'n_days_to_treatment_end'], inplace=True)
blood.rename(columns={feature: feature.replace('-', '_') for feature in blood.columns},
             inplace=True)

# Select study of interest
patient = st.sidebar.selectbox('Select patient of interest', lesions.gpcr_id.unique())
study = st.sidebar.selectbox('Select study phase of interest',
                             lesions[lesions.gpcr_id == patient].study_name.unique(), 0)

df = lesions[(lesions.study_name == study) & (lesions.gpcr_id == patient)].copy()
response = progression[(progression.study_name == study) & (progression.gpcr_id == patient)] \
    .prediction_score.item()

df.reset_index(inplace=True)
    
radiomics_features = ['vol_ccm', 'max_suv_val', 'mean_suv_val', 'min_suv_val', 'sd_suv_val']

sbm1, sbm2 = st.sidebar.columns(2)

sbm1.metric('No. of lesions', df.shape[0])
sbm2.metric('TMTV', f'{df.vol_ccm.sum():,.0f} ccm')
st.sidebar.metric('Response', response)


with st.expander('See lesion statistics:'):
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.histogram(df, x='assigned_organ', labels=figure_labels)
            .update_layout(legend=top_h_legend, yaxis_title='Lesions', width=WIDTH / 2)
            .update_xaxes(categoryorder='total descending')
        )

    with col2:
        st.plotly_chart(
            px.scatter_matrix(df[radiomics_features], labels=figure_labels)
            .update_layout(legend=top_h_legend, width=WIDTH / 2)
        )
        
    size = 1000
        
    ds = pd.DataFrame(
        np.vstack(
            (np.concatenate(
                [np.random.normal(r.mean_suv_val, r.sd_suv_val, size)
                 for _, r in df[['mean_suv_val', 'sd_suv_val']].iterrows()]),
             np.concatenate(
                [np.array([f'Lesion {lesion}'] * size) for lesion in range(df.shape[0])]))
        ).T, columns=['value', 'distribution'])

    ds['value'] = ds.value.astype(np.float32)

    st.plotly_chart(
        px.histogram(ds, x='value', color='distribution')
        .update_layout(barmode='overlay', legend=top_h_legend, width=WIDTH)
        .update_traces(opacity=0.5)
    )

con, col = st.columns(2)

connectivity = con.radio('Connectivity', ('fully', 'organ', 'distance', 'wasserstein'))
color_metric = col.selectbox('Color', [*radiomics_features, 'assigned_organ'])

if connectivity in ['distance', 'wasserstein']:
    dm, dt = st.columns(2)
    
    if connectivity == 'distance':
        distance_metric = dm.selectbox('Metric', radiomics_features)
    
    distance_threshold = dt.slider('Threshold', 0., 2., .2)

num_nodes = df.shape[0]
edge_index = []

# Connect lesions using different rhetorics
if connectivity == 'organ':
    for i in range(num_nodes):
        source = df.loc[i].assigned_organ
        targets = list(df[df.assigned_organ == source].index)

        edge_index.extend([[i, j] for j in targets if i != j])
        
elif connectivity == 'fully':
    edge_index = list(permutations(range(num_nodes), 2, ))
    
elif connectivity == 'distance':
    for i in range(num_nodes):
        source = df.loc[i][distance_metric]
        targets = list(df[abs(df[distance_metric] - source) < distance_threshold].index)

        edge_index.extend([[i, j] for j in targets if i != j])
        
elif connectivity == 'wasserstein':
    
    for i in range(num_nodes):
        source_mean, source_sd = df.loc[i].mean_suv_val, df.loc[i].sd_suv_val
        source_distribution = np.random.normal(source_mean, source_sd, 1000)
        
        targets = [id for id, mu, sd in zip(df.index, df.mean_suv_val, df.sd_suv_val)
                   if wasserstein_distance(source_distribution,
                                           np.random.normal(mu, sd, 1000)) < distance_threshold]

        edge_index.extend([[i, j] for j in targets if i != j])
    

else:
    raise ValueError(f'{connectivity} is not a possible connectivity method.')

edge_index = torch.tensor(edge_index).t()

x = np.array(
    preprocess(df[['gpcr_id', 'assigned_organ', *radiomics_features]],
               features_numerical=radiomics_features, features_categorical=['assigned_organ'])[0]
    .drop(columns='gpcr_id'))

y = int(response == 'PD')

graph = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, y=y)

m1, m2, m3, m4 = st.columns(4)

m1.metric('Nodes', graph.num_nodes)
m2.metric('Edges', graph.num_edges)
m3.metric('Avg. node deg.', f'{graph.num_edges / graph.num_nodes:.1f}')
m4.metric('Isolated nodes', graph.has_isolated_nodes())

G = to_networkx(graph, to_undirected=True)
pos = nx.kamada_kawai_layout(G)

fig, ax = plt.subplots()

if color_metric == 'assigned_organ':
    cmap = plt.cm.tab20
    organs = dict(zip(df[color_metric].unique(), range(len(df[color_metric].unique()))))
    node_color = [cmap(organs[organ]) for organ in df[color_metric]]
    
    # make empty plot with correct color and label for each group
    for organ, i in organs.items():
        plt.scatter([], [], c=cmap(i), label=organ)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2),
               ncol=round(len(organs) / 2))
    
    nx.draw_networkx(G, pos, node_color=node_color, cmap=cmap)
    
else:
    cmap = plt.cm.Blues
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(df[color_metric].min(), df[color_metric].max()))
    node_color = range(df.shape[0])

    nx.draw_networkx(G, pos, node_color=node_color, cmap=cmap)
    plt.colorbar(sm, label=figure_labels[color_metric])

st.pyplot(fig)
