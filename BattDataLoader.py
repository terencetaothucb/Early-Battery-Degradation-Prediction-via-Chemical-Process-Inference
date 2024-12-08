# -*- coding: utf-8 -*-
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import MultivariateNormal
import pandas as pd
import numpy as np
from scipy.signal import medfilt
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

# Raw and processed datasets have been deposited in TBSI-Sunwoda-Battery-Dataset
# The dataset can be accessed at https://github.com/terencetaothucb/TBSI-Sunwoda-Battery-Dataset.

class BattDataset(Dataset):
    def __init__(self, data, train):
        self.train = train
        self.train_data = []
        self.test_data = []
        T45 = data[data['T'] >= 0]#45
        T45 = T45[T45['T'] <= 1]
        T35 = data[data['T'] <= 0 ]
        T35 = T35[T35['T'] >= -1 ]
        filtered_data2 = data[data["T"] >= 1]
        filtered_data1 = data[data["T"] <= -1]
        T25 = filtered_data1
        T55 = filtered_data2

        self.train_data.append(T25)
        self.train_data.append(T55)
        self.train_data.append(T35[T35["cyc"] *1299 <= 200])
        self.train_data.append(T45[T45["cyc"]*1099 <= 200])
        self.train_data = pd.concat(self.train_data)

        self.test_data.append(T35[T35["cyc"] *1299 > 200])
        self.test_data.append(T45[T45["cyc"] *1099 > 200])
        self.test_data = pd.concat(self.test_data)

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]

    def __getitem__(self,idx):
        if self.train:
            # import pdb; pdb.set_trace()
            domain = self.train_data.iloc[idx][["T","U1","U2","U3","U4","U5","U6","U7","U8","U9","cyc"]].values
            feature = self.train_data.iloc[idx][10:52].values
            y = self.train_data.iloc[idx][["filter_cap"]].values
            y_plot = self.train_data.iloc[idx][["cap"]].values
            return domain, feature, y, y_plot
        else:
            domain = self.test_data.iloc[idx][["T","U1","U2","U3","U4","U5","U6","U7","U8","U9","cyc"]].values
            feature = self.test_data.iloc[idx][10:52].values
            y = self.test_data.iloc[idx][["filter_cap"]].values
            y_plot = self.test_data.iloc[idx][["cap"]].values
            return domain, feature, y , y_plot
