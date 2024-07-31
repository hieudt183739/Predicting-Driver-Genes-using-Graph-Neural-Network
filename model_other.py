import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.lin1 = Linear(128, 64)
        self.lin2 = Linear(64, 1)
    def forward(self, feature):

        x = feature 
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        
        return x

class CNN1(torch.nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.lin = Linear(4, 1)
    def forward(self, feature):
        x = feature 
        x = F.dropout(x, training=self.training)
        x = self.lin(x)

        return x


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.lin1 = Linear(19, 128)     # specific cancer,e.g.luad
        self.lin2 = Linear(128, 32)
        self.lin3 = Linear(32, 1)
    def forward(self, feature):
        x = F.dropout(feature, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, training=self.training)
        x = self.lin3(x)

        return x

