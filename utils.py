import torch
import torch.nn as nn

def mape_loss(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))
    
# MyNetwork1 is chemical process prediction model considering initial manufacturing variability (ChemicalProcessModel)
# Input U + cycles; Output features
class MyNetwork1(nn.Module):
    def __init__(self):
        super(MyNetwork1, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 42)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
# MyNetwork2 is battery degradation trajectory model : Input features + U + cycles; Output capacity
class MyNetwork2(nn.Module):
    def __init__(self):
        super(MyNetwork2, self).__init__()
        self.fc1 = nn.Linear(53, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# MyNetwork3 is chemical process prediction model considering both initial manufacturing variability and temperature.
# Input U + cycles + T; Output features
class MyNetwork3(nn.Module):
    def __init__(self):
        super(MyNetwork1, self).__init__()
        self.fc1 = nn.Linear(11, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 42)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


