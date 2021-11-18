import torch
import torch.nn.functional as F
from torch.nn import Linear, LogSoftmax, ModuleList, Sequential, BatchNorm1d, ReLU

from torch_geometric.nn import GATv2Conv as GATConv, GraphConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from models.custom import SparseModule


class BaselineGNN(SparseModule):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        node_features_dim: int,
        graph_features_dim: int,
        layer_type: str = 'GraphConv',
        num_layers: int = 5,
    ):
        super(BaselineGNN, self).__init__()
        self.layer_type = layer_type
        self.num_layers = num_layers
        
        self.convs = ModuleList()

        feature_extractor = \
            self.create_layer(in_channels=node_features_dim, out_channels=hidden_dim)

        self.convs.append(feature_extractor)
        
        for step in range(num_layers - 1):
            layer = \
                self.create_layer(in_channels=hidden_dim, out_channels=hidden_dim)
            
            self.convs.append(layer)

        self.fc1 = Linear(hidden_dim * 2 + graph_features_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

        self.readout = LogSoftmax(dim=-1)

    def forward(self, data):
        
        x, edge_index, batch, graph_features = \
            data.x, data.edge_index, data.batch, data.graph_features
        
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, edge_index))
        
        # Concatenate pooling from graph embeddings with graph features
        x = torch.cat(
            [gmp(x, batch), gap(x, batch), graph_features.reshape(batch.unique().size(0), -1)], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return self.readout(x)
    
    def create_layer(self, **kwargs):
        """Create layer based on type

        Args:
            type (str): layer type

        Raises:
            ValueError: if type is not accepted within framework

        Returns:
            nn.Module: layer
        """

        if self.layer_type == 'GraphConv':
            return GraphConv(**kwargs)
        elif self.layer_type == 'GAT':
            return GATConv(**kwargs)
        elif self.layer_type == 'GIN':
            node_features, dim = kwargs['in_channels'], kwargs['out_channels']
            return GINConv(Sequential(
                Linear(node_features, dim), BatchNorm1d(dim), ReLU(),
                Linear(dim, dim), ReLU()))
        else:
            raise ValueError(f'{self.layer_type} is not a valid layer type')
    
    def __str__(self) -> str:
        """Representation"""
        return f'Baseline GNN with {self.num_layers} {self.layer_type} layers'
