from torch.nn import Linear, LogSoftmax
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

from models.custom import NamedModule, SizeableModule


class GIN(NamedModule, SizeableModule):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        node_features_dim: int,
        edge_features_dim: int = None
    ):
        super(GIN, self).__init__()
        self.hidden_dim = hidden_dim

        self.conv1 = GINConv(node_features_dim, hidden_dim)
        self.conv2 = GINConv(hidden_dim, hidden_dim)
        self.conv3 = GINConv(hidden_dim, hidden_dim)
        self.conv4 = GINConv(hidden_dim, hidden_dim)
        self.conv5 = GINConv(hidden_dim, hidden_dim)

        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

        self.readout = LogSoftmax(dim=-1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return self.readout(x)
    
    def __str__(self) -> str:
        """Representation"""
        return "GIN Network"
