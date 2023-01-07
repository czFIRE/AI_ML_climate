import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime as dt
import matplotlib.pyplot as plt


class TimeSeriesDataset(Dataset):   
    def __init__(self, X, y, seq_len=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - self.seq_len

    def __getitem__(self, index):
        return self.X[index:index+self.seq_len], self.y[index+self.seq_len]

class TSModel_rnn(nn.Module):
    def __init__(self, n_features, n_hidden: int = 64, n_layers: int = 2):  
        super(TSModel_rnn, self).__init__()

        self.n_hidden = n_hidden
        self.lstm = nn.RNN(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers,
            dropout=0.5
        )
        self.linear = nn.Linear(n_hidden, 1)
        
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        lstm_out = hidden[-1]  # output last hidden state output
        y_pred = self.linear(lstm_out)
        
        return y_pred

class TSModel_lstm(nn.Module):
    def __init__(self, n_features, model_parameters: None|dict = {"dropout":0.5, "bias": True, "batch_first": True, "hidden_size": 64, "num_layers": 2}):
        super(TSModel_lstm, self).__init__()

        if (model_parameters is None):
            model_parameters = {"dropout":0.5, "bias": True, "batch_first": True, "hidden_size": 64, "num_layers": 2}
            
        self.n_hidden: int = model_parameters["hidden_size"]
        self.lstm = nn.LSTM(
            input_size=n_features,
            **model_parameters,
            
            # hidden_size=self.n_hidden,
            # batch_first=True,
            # num_layers=model_parameters["n_layers"],
            # dropout=0.5,
            # proj_size = 0,  
        )
        self.linear = nn.Linear(self.n_hidden, 1)
        
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        lstm_out = hidden[-1]  # output last hidden state output
        y_pred = self.linear(lstm_out)
        
        return y_pred

LSTM_Model = None
RNN_Model = None

# I'm disgusted by this but oh well
def reset_models():
    global LSTM_Model
    global RNN_Model

    LSTM_Model = None
    RNN_Model = None

def train_model(train: pd.DataFrame,
                test: pd.DataFrame,
                features: list[str],
                parameters:dict,
                model_name: str,
                model_parameters: None|dict = None,
                create_new_model = False,
                ):
    train_dataset = TimeSeriesDataset(np.array(train[features]), 
                    np.array(train["prec"]), seq_len=parameters["sequence_length"])
    train_loader = DataLoader(train_dataset, batch_size=parameters["batch_size"], shuffle=False)
    test_dataset = TimeSeriesDataset(np.array(test[features]),
                        np.array(train["prec"]), seq_len=parameters["sequence_length"])
    test_loader = DataLoader(test_dataset, batch_size=parameters["batch_size"], shuffle=False)

    n_features = train[features].shape[1]

    global LSTM_Model
    global RNN_Model

    # the old version created a new model for each file!!!
    if model_name == "LSTM":
        if (create_new_model or LSTM_Model is None):
            LSTM_Model = TSModel_lstm(n_features, model_parameters = model_parameters)
        model = LSTM_Model
    else:
        if (create_new_model or RNN_Model is None):
            RNN_Model = TSModel_rnn(n_features)
        model = RNN_Model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_hist = []
    test_hist = []

    best_loss = np.inf
    epochs_no_improve = 0
    for epoch in range(1, parameters["n_epochs"]+1):
        running_loss = 0
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            data = torch.Tensor(np.array(data)).to(device)
            
            output = model(data)
            loss = criterion(output.flatten(), target.type_as(output))
                # if type(criterion) == torch.nn.modules.loss.MSELoss:
                #     loss = torch.sqrt(loss)  # RMSE
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        running_loss /= len(train_loader)
        train_hist.append(running_loss)

            # test loss
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = torch.Tensor(np.array(data)).to(device)

                output = model(data)
                loss = criterion(output.flatten(), target.type_as(output))
                test_loss += loss.item()
            test_loss /= len(test_loader)
            test_hist.append(test_loss)

            # early stopping
            if test_loss < best_loss:
                best_loss = test_loss
                #torch.save(model.state_dict(), Path(model_dir, 'model.pt'))
                torch.save(model.state_dict(), f"{parameters['path']}{model_name}_model.pt")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve == parameters["n_epochs_stop"]:
                print("Early stopping.")
                break

        print(f'Epoch {epoch} train loss: {round(running_loss,4)} test loss: {round(test_loss,4)}')

        hist = pd.DataFrame()
        hist['training_loss'] = train_hist
        hist['test_loss'] = test_hist

    print("Completed.")
    
    return hist

def descale(descaler, values):
    values_2d = np.array(values)[:, np.newaxis]
    return descaler.inverse_transform(values_2d).flatten()

def print_loss_metrics(y_true, y_pred):
    print(mean_squared_error(y_true.tolist(), y_pred.tolist(), squared=False))

def predict(df: pd.DataFrame, parameters, features, orginal_scaler, model_name:str, model_parameters: None|dict = None):
    if model_name == "LSTM":
        model = TSModel_lstm(df[features].shape[1], model_parameters=model_parameters)
    else:
        model = TSModel_rnn(df[features].shape[1])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # We don't have the problem of the new model here since we're loading the best
    model.load_state_dict(torch.load(f"{parameters['path']}{model_name}_model.pt"))
    model.eval()

    test_dataset = TimeSeriesDataset(np.array(df[features]), np.array(df["prec"]), seq_len=parameters["sequence_length"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    labels = []

    with torch.no_grad():
            for features, target in test_loader:
                features = torch.Tensor(np.array(features)).to(device)
                output = model(features)
                predictions.append(output.item())
                labels.append(target.item())

    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = orginal_scaler.min_[0], orginal_scaler.scale_[0]
    predictions_descaled = descale(descaler, predictions)
    labels_descaled = descale(descaler, labels)

    return predictions_descaled, labels_descaled
