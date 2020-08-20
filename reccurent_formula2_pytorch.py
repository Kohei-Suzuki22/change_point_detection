import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optimizers
from torchsample.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import time
import statistics
import ipdb


class RNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.l1 = nn.RNN(1, hidden_dim,nonlinearity='tanh',batch_first=True)
        self.l2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.l1.weight_ih_l0)
        nn.init.orthogonal_(self.l1.weight_hh_l0)
    def forward(self, x):
        h, _ = self.l1(x)
        y = self.l2(h[:, -1])
        return y

if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    1. データの準備
    '''
    # def sin(x, T=100):
    #     box = np.array([])
    #     box = np.append(box,(np.sin(2.0 * np.pi * x[:100] / T)))
    #     box = np.append(box,(2 * np.sin(2.0 * np.pi * x[100:] / T)))
    #     return box

    # def toy_problem(T=100, ampl=0.05):
    #     x = np.arange(0, 2*T)
    #     noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    #     return sin(x) + noise

    def func(x,t,a=4):
        return a * x[t] * (1 - x[t])

    y = np.array([0.2])

    for t in range(500):
        if t < 250:
            y = np.append(y,func(y,t,3.7))
        else:
            y = np.append(y,func(y,t))

    # T = 100

    # f = toy_problem(T).astype(np.float32)
    length_of_sequences = len(y)
    affect_length = 1
    x = []
    t = []
    for i in range(length_of_sequences - affect_length):
        x.append(y[i:i+affect_length])
        t.append(y[i+affect_length])
    x = np.array(x).reshape(-1, affect_length, 1)
    t = np.array(t).reshape(-1, 1)
    '''
    2. モデルの構築
    '''

    model = RNN(50).to(device)
    '''
    3. モデルの学習
    '''
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optimizers.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True)


    def compute_loss(t, y):
        return criterion(y, t)
    def train_step(x, t):
        x = torch.Tensor(x).to(device)
        t = torch.Tensor(t).to(device)
        model.train()
        preds = model(x)
        loss = compute_loss(t, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, preds
    def val_step(x, t):
        x = torch.Tensor(x).to(device)
        t = torch.Tensor(t).to(device)
        model.eval()
        preds = model(x)
        loss = criterion(preds, t)
        return loss, preds
    epochs = 1000
    batch_size = 25

    n_batches = x.shape[0] // batch_size
    hist = {'loss': [], 'val_loss': []}
    all_loss = []
    loss_per_epochs = []
    es = EarlyStopping(patience=10, verbose=1)
    for epoch in range(epochs):
        train_loss = 0.
        val_loss = 0.


        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            loss, _ = train_step(x[start:end], t[start:end])
            all_loss.append(loss.item())
            train_loss += loss.item()
        loss_per_epochs.append(train_loss)
        train_loss /= n_batches
        hist['loss'].append(train_loss)
        print('epoch: {}, loss: {:.3}'.format(epoch+1,train_loss))
        # if es(val_loss):
        #     break
    '''
    4. モデルの評価
    '''
    model.eval()
    # sin波の予測
    # sin = toy_problem(T, ampl=0.)
    gen = [None for i in range(affect_length)]
    z = x[:1]
    for i in range(length_of_sequences - affect_length):
        z_ = torch.Tensor(z[-1:]).to(device)
        preds = model(z_).data.cpu().numpy()
        z = np.append(z, preds)[1:]
        z = z.reshape(-1, affect_length, 1)
        gen.append(preds[0, 0])
    # 予測値を可視化
    fig = plt.figure()
    plt.rc('font', family='serif')
    # plt.xlim([0, 2*T])
    plt.xlabel("t")
    plt.ylabel("y")
    # plt.ylim([-1.5, 1.5])
    plt.plot(range(len(y)), y,linestyle='--', linewidth=0.5,color="blue",label="row_data")
    plt.plot(range(len(y)), gen, linewidth=1,marker='o', markersize=1, markerfacecolor='black',markeredgecolor='black',color="red",label="pred")
    plt.legend(loc="lower left")
    plt.show()


    plt.xlabel("epochs")
    plt.ylabel("y")
    plt.xlim(0,1000)
    plt.plot(loss_per_epochs,color="blue",label="loss_per_epoch")
    plt.show()

    plt.xlabel("all_loss(per_batch)")
    plt.ylabel("y")
    plt.xlim(10000,11000)
    plt.plot(all_loss,color="blue",label="loss_per_batch(all_loss)")
    plt.show()
    ipdb.set_trace()


# lossが10万個取りたい。(batch10 * epochs 1000)