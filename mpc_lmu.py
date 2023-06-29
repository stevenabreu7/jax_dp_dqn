import pandas as pd
import os
import numpy as np
import torch

import numpy as np

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from scipy.signal import cont2discrete

# ------------------------------------------------------------------------------

def leCunUniform(tensor):
    """ 
        LeCun Uniform Initializer
        References: 
        [1] https://keras.rstudio.com/reference/initializer_lecun_uniform.html
        [2] Source code of _calculate_correct_fan can be found in https://pytorch.org/docs/stable/_modules/torch/nn/init.html
        [3] Yann A LeCun, Léon Bottou, Genevieve B Orr, and Klaus-Robert Müller. Efficient backprop. In Neural networks: Tricks of the trade, pages 9–48. Springer, 2012
    """
    fan_in = init._calculate_correct_fan(tensor, "fan_in")
    limit = np.sqrt(3. / fan_in)
    init.uniform_(tensor, -limit, limit) # fills the tensor with values sampled from U(-limit, limit)

# ------------------------------------------------------------------------------

class LMUCell(nn.Module):
    """ 
    LMU Cell

    Parameters:
        input_size (int) : 
            Size of the input vector (x_t)
        hidden_size (int) : 
            Size of the hidden vector (h_t)
        memory_size (int) :
            Size of the memory vector (m_t)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system
        learn_a (boolean) :
            Whether to learn the matrix A (default = False)
        learn_b (boolean) :
            Whether to learn the matrix B (default = False)
    """

    def __init__(self, input_size, hidden_size, memory_size, theta, learn_a = False, learn_b = False):
        
        super(LMUCell, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.f = nn.Tanh()

        A, B = self.stateSpaceMatrices(memory_size, theta)
        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

        if learn_a:
            self.A = nn.Parameter(A)
        else:
            self.register_buffer("A", A)
    
        if learn_b:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

        # Declare Model parameters:
        ## Encoding vectors
        self.e_x = nn.Parameter(torch.empty(1, input_size))
        self.e_h = nn.Parameter(torch.empty(1, hidden_size))
        self.e_m = nn.Parameter(torch.empty(1, memory_size))
        ## Kernels
        self.W_x = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_m = nn.Parameter(torch.empty(hidden_size, memory_size))

        self.initParameters()

    def initParameters(self):
        """ Initialize the cell's parameters """

        # Initialize encoders
        leCunUniform(self.e_x)
        leCunUniform(self.e_h)
        init.constant_(self.e_m, 0)
        # Initialize kernels
        init.xavier_normal_(self.W_x)
        init.xavier_normal_(self.W_h)
        init.xavier_normal_(self.W_m)

    def stateSpaceMatrices(self, memory_size, theta):
        """ Returns the discretized state space matrices A and B """

        Q = np.arange(memory_size, dtype = np.float64).reshape(-1, 1)
        R = (2*Q + 1) / theta
        i, j = np.meshgrid(Q, Q, indexing = "ij")

        # Continuous
        A = R * np.where(i < j, -1, (-1.0)**(i - j + 1))
        B = R * ((-1.0)**Q)
        C = np.ones((1, memory_size))
        D = np.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
            system = (A, B, C, D), 
            dt = 1.0, 
            method = "zoh"
        )
        
        return A, B

    def forward(self, x, state):
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, input_size]
            state (tuple): 
                h (torch.tensor) : [batch_size, hidden_size]
                m (torch.tensor) : [batch_size, memory_size]
        """

        h, m = state

        # Equation (7) of the paper
        u = F.linear(x, self.e_x) + F.linear(h, self.e_h) + F.linear(m, self.e_m) # [batch_size, 1]

        # Equation (4) of the paper
        m = F.linear(m, self.A) + F.linear(u, self.B) # [batch_size, memory_size]

        # Equation (6) of the paper
        h = self.f(
            F.linear(x, self.W_x) +
            F.linear(h, self.W_h) + 
            F.linear(m, self.W_m)
        ) # [batch_size, hidden_size]

        return h, m

# ------------------------------------------------------------------------------

class LMUModel(torch.nn.Module):
    """ A simple model for the psMNIST dataset consisting of a single LMU layer and a single dense classifier """

    def __init__(self, input_size, output_size, hidden_size, memory_size, theta, learn_a = False, learn_b = False):
        super(LMUModel, self).__init__()
        self.lmu = LMUCell(input_size, hidden_size, memory_size, theta, learn_a, learn_b)
        self.classifier = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = []
        h_0 = torch.zeros(x.shape[0], self.lmu.hidden_size)
        m_0 = torch.zeros(x.shape[0], self.lmu.memory_size)
        state = (h_0, m_0)
        for t in range(x.shape[1]):
            state = self.lmu(x[:,t,:], state) # [batch_size, hidden_size]
            output = self.classifier(state[0])
            out.append(output) # [batch_size, output_size]
        return torch.stack(out, dim=1) # [batch_size, seq_len, output_size]

# ------------------------------------------------------------------------------

def load_data(folder):
    X = []
    Y = []
    for fname in os.listdir(folder):
        df = pd.read_csv(f'{folder}/{fname}', skiprows=28)[:-1]
        x = df[[
            'angle_sin', 'angle_cos', 'angleD', 'position', 
            'positionD', 'target_equilibrium', 'target_position'
        ]]
        y = df[['Q']]
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def load_train_test_data():
    folder = 'DG-27s-and-1s500ms-noisy-u/Recordings/Train/Train-1s500ms'
    x_train, y_train = load_data(folder)
    folder = 'DG-27s-and-1s500ms-noisy-u/Recordings/Test/Test-1s500ms'
    x_test, y_test = load_data(folder)
    return x_train, y_train, x_test, y_test

# ------------------------------------------------------------------------------

def train_model(hidden_size, memory_size, theta, epochs=500):
    x_train, y_train, x_test, y_test = load_train_test_data()
    model = LMUModel(input_size=7, output_size=1, hidden_size=hidden_size, 
                     memory_size=memory_size, theta=theta)
    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.MSELoss()

    batch_size = 64
    n_batches = x_train.shape[0] // batch_size

    for epoch in range(epochs+1):
        # eval
        if epoch % 50 == 0:
            loss_train = []
            for batch_idx in range(n_batches-1):
                model.eval()
                x = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                y = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                ypr = model(x)
                loss_train.append(loss(ypr, y).item())
            loss_test = []
            for batch_idx in range(x_test.shape[0] // batch_size):
                model.eval()
                x = x_test[batch_idx*batch_size:(batch_idx+1)*batch_size]
                y = y_test[batch_idx*batch_size:(batch_idx+1)*batch_size]
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                ypr = model(x)
                loss_test.append(loss(ypr, y).item())
            print(epoch, 'train', np.array(loss_train).mean(), 'test', np.array(loss_test).mean())

        # train
        epoch_loss = []
        model.train()
        for batch_idx in range(n_batches-1):
            x = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
            y = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
            
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(x)
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()

            epoch_loss.append(l.item())

        # log training loss
        avg_epoch_loss = np.array(epoch_loss).mean()
        if epoch % 10 == 0:
            print(epoch, avg_epoch_loss)
        else:
            print(epoch, avg_epoch_loss, end='\r')

    model_name = f'mpc_models/lmu_7-1-{hidden_size}-{memory_size}-{theta}.pt'
    torch.save(model.state_dict(), model_name)

def load_model(model_name):
    hidden_size = int(model_name.split('-')[2])
    memory_size = int(model_name.split('-')[3])
    theta = float(model_name.split('-')[4].split('.')[0])
    model = LMUModel(input_size=7, output_size=1, hidden_size=hidden_size,
                     memory_size=memory_size, theta=theta)
    model.load_state_dict(torch.load(model_name))
    return model

def train_mlp(n_neurons=64, epochs=500):
    x_train, y_train, x_test, y_test = load_train_test_data()
    x_train = x_train.reshape(-1, 7)
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 7)
    y_test = y_test.reshape(-1, 1)

    model = torch.nn.Sequential(
        torch.nn.Linear(7, n_neurons),
        torch.nn.ReLU(),
        torch.nn.Linear(n_neurons, 1)
    )
    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.MSELoss()

    batch_size = 512
    n_batches = x_train.shape[0] // batch_size

    for epoch in range(epochs+1):
        # eval
        if epoch % 50 == 0:
            loss_train = []
            for batch_idx in range(n_batches-1):
                model.eval()
                x = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                y = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                ypr = model(x)
                loss_train.append(loss(ypr, y).item())
            loss_test = []
            for batch_idx in range(x_test.shape[0] // batch_size):
                model.eval()
                x = x_test[batch_idx*batch_size:(batch_idx+1)*batch_size]
                y = y_test[batch_idx*batch_size:(batch_idx+1)*batch_size]
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                ypr = model(x)
                loss_test.append(loss(ypr, y).item())
            print(epoch, 'train', np.array(loss_train).mean(), 'test', np.array(loss_test).mean())

        # train
        epoch_loss = []
        model.train()
        for batch_idx in range(n_batches-1):
            x = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
            y = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
            
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(x)
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()

            epoch_loss.append(l.item())

        # log training loss
        avg_epoch_loss = np.array(epoch_loss).mean()
        if epoch % 10 == 0:
            print(epoch, avg_epoch_loss)
        else:
            print(epoch, avg_epoch_loss, end='\r')

    model_name = f'mpc_models/mlp_7-1-{n_neurons}.pt'
    torch.save(model.state_dict(), model_name)
