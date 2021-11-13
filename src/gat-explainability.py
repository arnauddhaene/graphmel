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

filename = st.sidebar.selectbox("Please select a dataset", files, index=2)
infile = open(os.path.join(CHECKPOINTS_DIR, filename), 'rb')

dataset_train, dataset_test = pickle.load(infile)

# Display dataset metrics
train_metric, test_metric = st.sidebar.columns(2)

train_metric.metric('Train set size', len(dataset_train))
test_metric.metric('Test set size', len(dataset_test))

# Select datapoint
all_graphs = [*dataset_train, *dataset_test]
idx = st.select_slider("Please select a datapoint by index", range(len(all_graphs)))
graph = all_graphs[idx]

X_train, _, _, _ = preprocess(*fetch_data())

color_metric = st.selectbox('Color_metric', enumerate(X_train.columns), format_func=lambda kv: kv[1])

# Fetch model
model_args = dict(num_classes=2, hidden_dim=64, node_features_dim=graph.x.shape[1])

model = BaselineGNN(layer_type='GAT', **model_args)

# Wasserstein with distance 0.5 connected
storage_path = os.path.join(ASSETS_DIR, 'models/',
                            'Baseline GNN with 5 GAT layers-2021-11-12 16:22:02.947209.pkl')

# Fully connected model
# storage_path = os.path.join(ASSETS_DIR, 'models/',
#                             'Baseline GNN with 5 GAT layers-2021-11-13 17:02:37.658111.pkl')

model.load_state_dict(torch.load(storage_path))


def get_attention_weights(model: torch.nn.Module, graph: Data) -> torch.Tensor:
    
    x, edge_index = graph.x, graph.edge_index

    for step in range(len(model.convs) - 1):
        x = model.convs[step](x, edge_index)

    x, (edge_index, alpha) = model.convs[-1](x, edge_index, return_attention_weights=True)
    
    return edge_index, alpha


# Extract learned edge weights
edge_index, alpha = get_attention_weights(model, graph)
non_self_loop_idx = np.array(list(map(len, map(set, edge_index.t().tolist())))) > 1
edge_index, alpha = edge_index[:, non_self_loop_idx], alpha[non_self_loop_idx]

# Compute adjacency matrix of learned graph structure
A = np.zeros((graph.num_nodes, graph.num_nodes))

for (i, j), a in zip(edge_index.t(), alpha):
    A[i, j] = a

# Display graphs
fig, ax = plt.subplots(1, figsize=(8, 3))

G_ = to_networkx(Data(edge_index=edge_index))
pos_ = nx.kamada_kawai_layout(G_)

node_colors = graph.x[:, color_metric[0]]
node_cmap = plt.cm.Blues
node_sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=plt.Normalize(node_colors.min(), node_colors.max()))

edge_cmap = plt.cm.Reds
# edge_color != alpha because networkx reorders edges by sorting node no.
# because of this, we need to use G_.edges() to reorder alpha accordingly
edge_colors = [A[i, j] for i, j in map(list, G_.edges())]
edge_sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(min(edge_colors), max(edge_colors)))

nx.draw_networkx(G_, pos_, width=2.0, ax=ax,
                 node_color=node_colors, cmap=node_cmap,
                 edge_color=edge_colors, edge_cmap=edge_cmap)

plt.colorbar(node_sm, label=' '.join(X_train.columns[color_metric[0]].split('_')).capitalize())
plt.colorbar(edge_sm, label=r'Edge weight')

st.pyplot(fig)

# Display adjacency matrices
fig, ax = plt.subplots(1, 2, figsize=(8, 3))

sns.heatmap(nx.adjacency_matrix(to_networkx(graph)).todense(), cbar=False, ax=ax[0])
ax[0].set_title(r'$A$ from SUV similarity')

sns.heatmap(A, cmap=plt.cm.Reds, ax=ax[1])
ax[1].set_title(r'Learned $A$ from last GAT layer')

st.pyplot(fig)
