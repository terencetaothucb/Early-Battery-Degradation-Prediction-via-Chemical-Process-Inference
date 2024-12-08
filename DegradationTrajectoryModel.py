# -*- coding: utf-8 -*-
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn.functional as F
import warnings
from utils import *
from BattDataLoader import BattDataset
warnings.filterwarnings("ignore")

# a set of learning rate
learning_rates = [1e-3, 2e-3, 3e-3]
lr_losses = {}
# The information of the best model
best_lr = None
best_loss = float('inf')
best_model_state = None

train_epochs = 100
raw_data = pd.read_csv("./raw_data_0920.csv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create training dataset and its data loader with batch size 1 and shuffle enabled.
train_dataset = BattDataset(raw_data, train=True)  
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  
# Create validation dataset and its data loader with batch size 1 and no shuffle.
valid_dataset = BattDataset(raw_data, train=True)  
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)  
# Create test dataset and its data loader with batch size 1 and no shuffle.
test_dataset = BattDataset(raw_data, train=False)  
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  
# Define MSE loss function and move it to the device.
criterion = nn.MSELoss().to(device)  

for lr in learning_rates:

    logging.basicConfig(filename=f'./log/model_2_LR={lr}.log', level=logging.INFO, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    model_2 = MyNetwork2().to(device)
    optimizer2 = optim.Adam(model_2.parameters(), lr=0.0001, amsgrad=False)
    criterion = nn.MSELoss().to(device)
    l1_strength = 0.00001

    def add_l1_regularization(model, l1_strength):
        l1_regularization = torch.tensor(0.).to(device)
        for param in model.parameters():
            l1_regularization += torch.norm(param, p=1)
        return l1_strength * l1_regularization

    def loss_fn(outputs, labels, model, l1_strength):
        loss = criterion(outputs, labels)
        l1_regularization = add_l1_regularization(model, l1_strength)
        loss += l1_regularization
        return loss

    def mse(y_true, y_pred):
        return F.mse_loss(y_pred, y_true)

    # train
    for epoch in range(train_epochs):
        train_losses = []
        for batch, (domain, feature, y, y_plot)  in enumerate(train_loader):
            domain = domain.to(device)
            feature = feature.to(device)
            y = y.to(device).float()
            y_plot = y_plot.to(device).float()

            tensor2 = torch.tensor(feature, dtype=torch.float64).view(1, -1)
            y_tensor = torch.cat([domain,tensor2],dim= 1)
            model2_output = model_2(y_tensor.float()).float()
            feature = feature.float()
            loss2 = loss_fn(model2_output, y_plot, model_2, l1_strength)

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            # print the process of training
            if batch % 1000 == 0:
                logging.info('Epoch [{}/{}], Loss: {:.4f}'.format(batch, train_epochs, loss1.item()))

            train_losses.append(loss2.item())

        # calculate the average loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        print('train loss:',avg_train_loss)
        logging.info(f'Epoch {epoch+1}/{train_epochs} - Train Loss: {avg_train_loss:.4f}')

        # test on validation set
        test_loss = 0.0
        total_batches = 0
        true = []
        pred = []
        test_losses = []

        for batch, (domain, feature, y, y_plot) in enumerate(test_loader):
            domain = domain.to(device)
            feature = feature.to(device)
            y = y.to(device).float()
            y_plot = y_plot.to(device)
            tensor2 = torch.tensor(feature, dtype=torch.float64).view(1, -1)
            y_tensor = torch.cat([domain,tensor2],dim= 1)
            feature_pred = model_2(y_tensor.float().to(device))
            true.append(y_plot.detach().cpu().numpy())
            pred.append(feature_pred.detach().cpu().numpy())

            batch_loss = mape_loss(y, feature_pred)
            test_losses.append(batch_loss.item())

        avg_val_loss = sum(test_losses) / len(test_losses)
        print('test loss:',avg_val_loss)

        lr_losses[lr] = avg_val_loss
        logging.info(f'Learning Rate: {lr}, Epoch: {epoch} , Validation Loss: {avg_val_loss:.4f}')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_save_path = f"./best_model_2_lr_{best_lr}_{epoch}.pt"

        # renew the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_lr = lr
            best_model_state = model_2.state_dict()
            logging.info(f"Model saved to {model_save_path}")
            # save the best model
            torch.save(best_model_state, f"./best_model_2_lr_{best_lr}_{epoch}.pt")
            logging.info(f'Best Learning Rate: {best_lr}, Best epoch: {epoch}, Best Validation Loss: {best_loss:.4f}')

