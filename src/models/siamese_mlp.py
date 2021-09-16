import torch
import torch.nn as nn
from models.custom import SizeableModule, NamedModule, WeightInitializableModule


class SiameseMLP(SizeableModule, NamedModule, WeightInitializableModule):
    """
    Siamese Multi Layer Perceptron

    Attributes:
        fc1 (nn.Linear)       : first fully connected layer
        fc2 (nn.Linear)       : second fully connected layer
        fc3 (nn.Linear)       : third fully connected layer
        fc4 (nn.Linear)       : last fully connected layer
        drop (nn.Dropout)     : dropout function
        sigmoid (nn.Sigmoid)  : sigmoid activation function
    """
    
    def __init__(self):
        """Constructor"""
        super().__init__()
        self.fc1 = nn.Linear(14 * 14, 128)
        self.fc2 = nn.Linear(128, 98)
        self.fc3 = nn.Linear(98, 49)
        self.fc4 = nn.Linear(49, 10)
        
        self.classifier = nn.Linear(20, 1)
        
        # dropout layer
        self.drop = nn.Dropout(0.2)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self.apply(self.weights_init)
        
    def forward_once(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass function used in the sub-network

        Args:
            x [float32]: input image with dimension Bx1x14x14 (for batch size B)

        Returns:
            [float32]: non activated tensor of dimension Bx1x10
        """

        # flatten image input
        x = x.flatten(start_dim=1)
        # add hidden layer, with relu activation function
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        
        x = self.relu(self.fc3(x))
        x = self.drop(x)
        
        x = self.fc4(x)

        return x

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass function for the global siamese CNN

        Args:
            x [float32]: input images with dimension Bx2x14x14 (for batch size B)

        Returns:
            [int]: predicted probability ]0,1[
            [float32] : predicted classe by pair, size Bx2x10
        """
        input1 = x[:, 0, :, :].view(-1, 1, 14, 14)  # size Bx1x14x14
        input2 = x[:, 1, :, :].view(-1, 1, 14, 14)
        
        x1 = self.forward_once(input1)  # size Bx1x10
        x2 = self.forward_once(input2)
        
        auxiliary = torch.stack((x1, x2), 1)  # size Bx2x10
        
        output = torch.cat((x1, x2), 1)  # size Bx1x20
        
        # output = self.relu(self.fc3(output))  # size Bx1x10
        output = self.sigmoid(self.classifier(output))  # size Bx1x1
        
        return output.squeeze(), auxiliary
    
    def __str__(self) -> str:
        """Representation"""
        return "Siamese Multi-Layer Perceptron"
