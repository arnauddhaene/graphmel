import os
import pickle

import numpy as np
import networkx as nx

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio

from utils import fetch_data, ASSETS_DIR, CHECKPOINTS_DIR, preprocess
from models import BaselineGNN


CONNECTION_DIR = '/Users/arnauddhaene/Downloads/'

pio.templates.default = 'seaborn'
top_h_legend = dict(orientation='h', yanchor="bottom", y=1.02)
figure_labels = {
    'assigned_organ': 'Assigned organ', 'vol_ccm': 'Volume (ccm)',
}
WIDTH = 900
HEIGHT = 400

st.set_page_config(
    layout="centered",
    page_icon="https://pbs.twimg.com/profile_images/1108310620590559232/7fTy3YtS_400x400.png")

# Styles
with open(os.path.join(ASSETS_DIR, "style.css")) as file:
    styles = file.read()
st.write(f"<style> {styles} </style>", unsafe_allow_html=True)

st.title("Graph Structure Learning using trained GAT")

# Select dataset
files = os.listdir(CHECKPOINTS_DIR)

filename = st.sidebar.selectbox("Please select a dataset", files, index=1)
infile = open(os.path.join(CHECKPOINTS_DIR, filename), 'rb')

dataset_train, dataset_test = pickle.load(infile)

train_metric, test_metric = st.sidebar.columns(2)

train_metric.metric('Train set size', len(dataset_train))
test_metric.metric('Test set size', len(dataset_test))

# Select datapoint
all_graphs = [*dataset_train, *dataset_test]
idx = st.select_slider("Please select a datapoint by index", range(len(all_graphs)))

graph = all_graphs[idx]

G = to_networkx(graph, to_undirected=True)
pos = nx.kamada_kawai_layout(G)

fig, ax = plt.subplots(1, 2, figsize=(8, 3))

X_train, _, _, _ = preprocess(*fetch_data())

color_metric = st.selectbox('Color_metric', enumerate(X_train.columns),
                            format_func=lambda kv: kv[1])

colors_ = graph.x[:, color_metric[0]]
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                           norm=plt.Normalize(colors_.min(), colors_.max()))

nx.draw_networkx(G, pos, node_color=colors_, cmap=plt.cm.Blues, ax=ax[0])
plt.colorbar(sm, label=X_train.columns[color_metric[0]], ax=ax[0])

model_args = dict(num_classes=2, hidden_dim=64, node_features_dim=graph.x.shape[1])

model = BaselineGNN(layer_type='GAT', **model_args)

storage_path = os.path.join(ASSETS_DIR, 'models/',
                            'Baseline GNN with 5 GAT layers-2021-11-12 16:22:02.947209.pkl')

model.load_state_dict(torch.load(storage_path))


def get_attention_weights(model: torch.nn.Module, graph: Data) -> torch.Tensor:
    
    x, edge_index = graph.x, graph.edge_index

    for step in range(len(model.convs) - 1):
        x = model.convs[step](x, edge_index)

    x, (edge_index, alpha) = model.convs[-1](x, edge_index, return_attention_weights=True)
    
    return edge_index, alpha


edge_index, alpha = get_attention_weights(model, graph)

non_self_loop_idx = np.array(list(map(len, map(set, edge_index.t().tolist())))) > 1

edge_index, alpha = edge_index[:, non_self_loop_idx], alpha[non_self_loop_idx]

G_ = to_networkx(Data(edge_index=edge_index))
pos_ = nx.kamada_kawai_layout(G_)

cmap = plt.cm.Reds
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(alpha.min(), alpha.max()))

nx.draw_networkx(G_, pos_, width=2.0,
                 edge_color=alpha.flatten().reshape(-1).tolist(),
                 edge_cmap=cmap, ax=ax[1])

plt.colorbar(sm, label='Learned edge weight')

st.pyplot(fig)

# Adjacency matrices
A = np.zeros((graph.num_nodes, graph.num_nodes))

for (i, j), a in zip(graph.edge_index.t(), alpha):
    A[i, j] = a

fig, ax = plt.subplots(1, 2, figsize=(8, 3))

sns.heatmap(nx.adjacency_matrix(G).todense(), cmap=cmap, ax=ax[0])

sns.heatmap(A, cmap=cmap, ax=ax[1])

st.pyplot(fig)
