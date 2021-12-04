import torch
# from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, LogSoftmax, ModuleList, Sequential, BatchNorm1d, ReLU, LSTM

from torch_geometric.nn import GATv2Conv as GATConv, GraphConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from models.custom import SparseModule


class BaselineGNN(SparseModule):
    def __init__(self, lesion_features_dim: int, hidden_dim: int,
                 layer_type: str = 'GraphConv', num_layers: int = 10):
        super(BaselineGNN, self).__init__()
        
        self.layer_type = layer_type
        self.num_layers = num_layers
        
        self.convs = ModuleList()

        feature_extractor = \
            self.create_layer(in_channels=lesion_features_dim, out_channels=hidden_dim)

        self.convs.append(feature_extractor)
        
        for step in range(num_layers - 1):
            layer = \
                self.create_layer(in_channels=hidden_dim, out_channels=hidden_dim)
            
            self.convs.append(layer)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:

        # Wasserstein edge weights are added if GraphConv layers are used
        if self.layer_type == 'GNN':
            conv_kwargs = dict(edge_weight=edge_weight)
        # if self.layer_type == 'GAT':
            # conv_kwargs = dict(edge_attr=edge_weight)
        else:
            conv_kwargs = dict()
        
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, edge_index, **conv_kwargs))
        
        return x
        
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
            # return GATConv(heads=3, edge_dim=1, **kwargs)
            return GATConv(heads=1, **kwargs)
        elif self.layer_type == 'GIN':
            node_features, dim = kwargs['in_channels'], kwargs['out_channels']
            return GINConv(Sequential(
                Linear(node_features, dim), BatchNorm1d(dim), ReLU(),
                Linear(dim, dim), ReLU()))
        else:
            raise ValueError(f'{self.layer_type} is not a valid layer type')
        

class TimeGNN(SparseModule):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        lesion_features_dim: int,
        study_features_dim: int,
        patient_features_dim: int,
        layer_type: str = 'GraphConv',
        num_layers: int = 10,
    ):
        super(TimeGNN, self).__init__()
        
        self.layer_type = layer_type
        self.num_layers = num_layers
        
        self.gnn = BaselineGNN(lesion_features_dim=lesion_features_dim, hidden_dim=hidden_dim,
                               layer_type=layer_type, num_layers=num_layers)

        self.rnn = LSTM(input_size=(hidden_dim * 2 + study_features_dim),
                        hidden_size=hidden_dim).to(dtype=torch.float64)
        
        self.fc1 = Linear(hidden_dim + patient_features_dim, hidden_dim).to(dtype=torch.float64)
        self.fc2 = Linear(hidden_dim, num_classes).to(dtype=torch.float64)

        self.readout = LogSoftmax(dim=-1).to(dtype=torch.float64)

    def forward(self, data):
        
        study_features, patient_features = data.study_features, data.patient_features
        
        xes = list(data.x.split(tuple(data.graph_sizes)))
        edge_indices = list(data.edge_index.split(tuple(data.split_sizes), dim=1))
        batches = list(data.batch.split(tuple(data.graph_sizes)))
        
        study_embeddings = []
        
        for i, (x, edge_index, batch) in enumerate(zip(xes, edge_indices, batches)):
            
            pooled_lesions_features = [gmp(self.gnn(x, edge_index), batch),
                                       gap(self.gnn(x, edge_index), batch)]
                
            # Size of which will be (len(xes), hidden_dim * 3)
            study_embeddings.append(torch.cat([
                *pooled_lesions_features, study_features[i, :].reshape(1, -1)
            ], dim=1).t())
        
        # len(xes) here is the sequence length
        rnn_output, _ = self.rnn(torch.cat(study_embeddings, dim=1).t().reshape(len(xes), 1, -1))
        study_pooled = rnn_output[-1, :, :].flatten()
        
        patient_pooled = torch.cat([study_pooled, patient_features], dim=0)

        patient_pooled = F.relu(self.fc1(patient_pooled)).to(dtype=torch.float64)
        patient_pooled = F.dropout(patient_pooled, p=0.25, training=self.training).to(dtype=torch.float64)
        patient_pooled = self.fc2(patient_pooled)

        return self.readout(patient_pooled).reshape(1, -1)
    
    def __str__(self) -> str:
        """Representation"""
        return f'TimeGNN with {self.num_layers} {self.layer_type} layers'
