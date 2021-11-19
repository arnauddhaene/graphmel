from typing import List

import torch
from torch.nn import Linear, LogSoftmax, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import DenseGraphConv, dense_diff_pool

from models.custom import DenseModule


class DiffPool(DenseModule):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        graph_features_dim: int,
        node_features_dim: int,
        num_nodes: List[int] = [50, 5]
    ):
        
        super(DiffPool, self).__init__()
        
        self.initial_pool = DenseGNN(node_features_dim, hidden_dim, num_nodes[0])
        self.initial_embed = DenseGNN(node_features_dim, hidden_dim, hidden_dim)
        
        self.pools = ModuleList()
        self.embeds = ModuleList()
        
        for nodes in num_nodes[1:]:
            self.pools.append(DenseGNN(hidden_dim, hidden_dim, nodes))
            self.embeds.append(DenseGNN(hidden_dim, hidden_dim, hidden_dim))
                
        self.embeds.append(DenseGNN(hidden_dim, hidden_dim, hidden_dim))
        
        self.fc1 = Linear(hidden_dim + graph_features_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

        self.readout = LogSoftmax(dim=-1)

    def forward(self, data):
        
        x, adj, mask, graph_features = data.x, data.adj, data.mask, data.graph_features
        
        s = self.initial_pool(x, adj, mask)
        x = self.initial_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        
        for step in range(len(self.pools)):
            s = self.pools[step](x, adj)
            x = self.embeds[step](x, adj)

            x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        
        x = self.embeds[-1](x, adj)
        
        x = x.mean(dim=1)
        
        x = torch.cat([x, graph_features], dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=-1)
    
    def __str__(self) -> str:
        """Representation"""
        return "DiffPool with DenseGraphConv GNN"


class DenseGNN(DenseModule):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        conv_layers: int = 5
    ):
        super(DenseGNN, self).__init__()
        
        self.convs = ModuleList()

        self.convs.append(DenseGraphConv(in_channels, hidden_dim))

        for _ in range(conv_layers - 1):
            self.convs.append(DenseGraphConv(hidden_dim, hidden_dim))

    def forward(self, x, adj, mask=None):
        
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
        
        return x
