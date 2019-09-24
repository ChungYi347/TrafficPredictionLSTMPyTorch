from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import csv, os
import numpy as np
from numpy.random import seed
seed(1)

import pandas as pd

import torch
import torch.nn as nn
torch.manual_seed(1)
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler
import argparse
import yaml
from time import gmtime, strftime

from metrics import *

def create_dataset(dataset, look_back=1, forward=1):
    # Data Preprocessing Function
    # look_back : Previous time step ex) look_back : 5 --> t-4 to t
    # forward : Target prediction time step ex) forward : 3 --> after 15minutes 
    
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1-forward):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back + forward - 1, :])
    return np.array(dataX), np.array(dataY)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_unit, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_unit = hidden_unit
        self.layer_dim = layer_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_unit, layer_dim).cuda()
        self.fc = nn.Linear(hidden_unit, output_dim).cuda()
    
    def forward(self, x):
        x = torch.transpose(x, 1, 0)
        hidden = (torch.randn(self.layer_dim, x.shape[1], self.hidden_unit).cuda(), torch.randn(self.layer_dim, x.shape[1], self.hidden_unit).cuda())
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output[-1, :, :])
        return output

class LSTMModel2(nn.Module):
    def __init__(self, input_dim, hidden_unit, layer_dim, output_dim):
        super(LSTMModel2, self).__init__()
        self.hidden_unit = hidden_unit
        self.layer_dim = layer_dim
        
        self.layers = nn.ModuleList([nn.LSTMCell(input_dim if l == 0 else hidden_unit, hidden_unit).cuda() for l in range(layer_dim)])
        self.layers.append(nn.Linear(hidden_unit, output_dim).cuda())
    
    def forward(self, x):
        x = torch.transpose(x, 1, 0)
        hidden = (torch.randn(x.shape[1], self.hidden_unit).cuda(), torch.randn(x.shape[1], self.hidden_unit).cuda())
        hc = [hidden for i in range(self.layer_dim)]

        for t in range(x.shape[0]):
            for i in range(self.layer_dim):
                if i == 0:
                    hc[i] = self.layers[i](x[t], hc[i])
                else:
                    hc[i] = self.layers[i](hc[i-1][0], hc[i])
        output = self.layers[-1](hc[-1][0])
        return output

class LSTMModel3(nn.Module):
    def __init__(self, input_dim, hidden_unit, layer_dim, output_dim):
        super(LSTMModel3, self).__init__()
        self.hidden_unit = hidden_unit
        self.layer_dim = layer_dim
        
        self.wi = nn.ParameterList([nn.Parameter(torch.Tensor(input_dim if i == 0 else hidden_unit, hidden_unit * 4).cuda()) for i in range(layer_dim)])
        self.hi = nn.ParameterList([nn.Parameter(torch.Tensor(hidden_unit, hidden_unit * 4).cuda()) for i in range(layer_dim)])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(hidden_unit * 4).cuda()) for i in range(layer_dim)])

        self.fc = nn.Linear(hidden_unit, output_dim).cuda()

        for p in self.parameters():
            if p.ndimension() < 2:
                nn.init.zeros_(p)
            else:
                nn.init.xavier_uniform_(p)

    def forward(self, x, init_states = None):
        x = torch.transpose(x, 1, 0)
        hidden = (torch.randn(x.shape[1], self.hidden_unit).cuda(), torch.randn(x.shape[1], self.hidden_unit).cuda())
        hc = [hidden for i in range(self.layer_dim)]

        hu = self.hidden_unit
        for t in range(x.shape[0]):
            for i in range(self.layer_dim):
                x_t = None
                if i == 0:
                    x_t = x[t]
                else:
                    x_t = h_t
                gates = x_t @ self.wi[i] + hc[i][0] @ self.hi[i] + self.bias[i]
                i_t, f_t, g_t, o_t = torch.sigmoid(gates[:, :hu]), torch.sigmoid(gates[:, hu:hu*2]), torch.tanh(gates[:, hu*2:hu*3]), torch.sigmoid(gates[:, hu*3:])
                c_t = f_t * hc[i][1] + i_t * g_t
                h_t = o_t * torch.tanh(c_t)
                hc[i] = (h_t, c_t)
        output = self.fc(hc[-1][0])
        return output

class trafficDataset(Dataset):
    def __init__(self, x, y): 
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

def main(args):
    with open(args.config_filename) as f:
        print("--Configureation--")
        config = yaml.load(f)
        print(config)
        file_pre = config['input']
        look_back = config['lookback']
        forward = config['forward']
        save = config['save']
        layer_dim = config['layerdim']
        hidden_unit = config['hiddenunit']
        input_file = config['input']
        output_file = config['output']
        epochs = config['epochs']
        batch_size = config['batch']
        learning_rate = config['learningrate']
        dic = {}
        X, y = [], []
        
        prev_sensor, current_sensor = [], []

        print("--Data Loading--")
        # Read input file
        df = pd.read_hdf(input_file)

        #print(df.columns[:-50])
        #df = df.drop(columns=list(df.columns[:-50]))

        # Normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(df.values)
        X, y = create_dataset(dataset, look_back=look_back, forward=forward)
        print(np.array(X).shape, np.array(y).shape)

        # Split train and test    
        size = int(len(X) * 0.8)
        X, y = torch.tensor(X, dtype=torch.float).cuda(), torch.tensor(y, dtype=torch.float).cuda()
        X_train, X_test = X[:size], X[size:] 
        y_train, y_test = y[:size], y[size:] 
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        y_test = scaler.inverse_transform(y_test.cpu().detach().numpy())
        train_dataset = trafficDataset(X_train, y_train)
        #test_dataset = trafficDataset(X_test, y_test)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        #test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        print("--Model Building--")
        # Build LSTM model
        # layer_dim : number of LSTM layers
        #model = mLSTMModel(X.shape[-1], hidden_unit, layer_dim, X.shape[-1])
        model = LSTMModel3(X.shape[-1], hidden_unit, layer_dim, X.shape[-1])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()


        for t in range(epochs):
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                y_pred = model(inputs)
                
                loss = loss_fn(y_pred, labels)
                #optimizer.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("epochs:", t, ", loss:", loss)

        # Prediction result and denormalization
        y_pred = model(X_test).cpu().detach().numpy()
        y_pred = scaler.inverse_transform(y_pred)
        print(y_pred)
        print(y_test)

        print("--Evaluating and Saving--")
        # Regression Metirc
        print("MAE : %s, RMSE : %s, MAPE : %s" % (masked_mae_np(y_pred, y_test, 0), masked_rmse_np(y_pred, y_test, 0), masked_mape_np(y_pred, y_test, 0)))

        # Save model and output
        if not os.path.exists('model'):
            os.makedirs('model')
        if not os.path.exists('output'):
            os.makedirs('output')

        np.save(output_file, y_pred)

        if save: 
            path = 'model/%s-%s-%s-%s-%s.json' % (strftime("%Y-%m-%d%H:%M:%S", gmtime()), str(look_back), str(forward), str(layer_dim), str(hidden_unit))
            torch.save(model, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='config/test.yaml', type=str, help='config filename')
    args = parser.parse_args()
    main(args)
