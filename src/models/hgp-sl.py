from typing import List

import torch
import torch.nn.functional as F
from torch.nn import Linear, LogSoftmax

from torch_geometric.nn import DenseGraphConv, dense_diff_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_adj, to_dense_batch

from models.custom import SparseModule


class HGP_SL(SparseModule):
    def __init__(
        self, num_classes: int, hidden_dim: int, node_features_dim: int,
        max_num_nodes: int, num_nodes: List[int] = [50, 5],
    ):
        
        super(HGP_SL, self).__init__()
        
        self.max_num_nodes = max_num_nodes
        
        self.pool1 = DenseGraphConv(node_features_dim, hidden_dim)
        self.embed1 = DenseGraphConv(node_features_dim, hidden_dim)
        
        self.pool2 = DenseGraphConv(hidden_dim, hidden_dim)
        self.embed2 = DenseGraphConv(hidden_dim, hidden_dim)
                
        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

        self.readout = LogSoftmax(dim=-1)

    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        adj = to_dense_adj(edge_index, batch, self.max_num_nodes)
        dense_x, mask = to_dense_batch(x, batch, self.max_num_nodes)
        
        s = F.relu(self.pool1(dense_x, adj, mask))
        x = F.relu(self.embed1(dense_x, adj, mask))

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        s = F.relu(self.pool2(x, adj))
        x = F.relu(self.embed2(x, adj))

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        
        x = self.embeds[-1](x, adj)
        
        x = x.mean(dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2
    
    def __str__(self) -> str:
        """Representation"""
        return "DiffPool with DenseGraphConv GNN"
