# 漸化式 x(t+1) = ax(t) (1-x(t))の時系列解析。
# ノイズなし。

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





affect_length = 1
np.random.seed(0) #乱数の固定

# 波形の生成

def func(x,t,a=4):
  return a * x[t] * (1 - x[t])

y = np.array([0.2])

# for t in range(499):
#   if t < 300:
#     y = np.append(y,func(y,t))
#   else:
#     y = np.append(y,func(y,t,5))

for t in range(300):
  y = np.append(y,func(y,t))
for t in range(300,500):
  y = np.append(y,func(y,t,3))





# print(y.shape)

affect_length = 1


def make_dataset(y, affect_length):
  factors = np.array([[]])
  answers = np.array([])
  for i in range(len(y) - affect_length):
    factors = np.append(factors,y[i:i+affect_length])
    answers = np.append(answers,y[i+affect_length])

  return(factors, answers)

(factors, answers) = make_dataset(y,affect_length)

factors = factors.reshape(-1,affect_length,1)
# print(factors.shape)
# print(answers.shape)
# print(factors.shape)      #: (268,32,1)
# print(answers.shape)      #: (268,)

n_in = 1
n_middle = 1
n_out = 1
n_hidden_unit = 100
lr = 0.0001


class RNN(nn.Module):

    np.random.seed(0) #乱数の固定
    affect_length = 1


# 波形の生成

    def func(x,t,a=4):
      return a * x[t] * (1 - x[t])

    y = np.array([0.2])

    # for t in range(499):
    #   if t < 300:
    #     y = np.append(y,func(y,t))
    #   else:
    #     y = np.append(y,func(y,t,5))

    for t in range(300):
      y = np.append(y,func(y,t))
    for t in range(199):
      y = np.append(y,func(y,3))

    # print(y.shape)

    def make_dataset(y, affect_length):
      factors = np.array([[]])
      answers = np.array([])
      for i in range(len(y) - affect_length):
        factors = np.append(factors,y[i:i+affect_length])
        answers = np.append(answers,y[i+affect_length])

      return(factors, answers)

    (factors, answers) = make_dataset(y,affect_length)

    factors = factors.reshape(-1,affect_length,1)

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

    length_of_sequences = len(y)

    x = []
    t = []

    for i in range(length_of_sequences - affect_length):
        x.append(y[i:i+affect_length])
        t.append(y[i+affect_length])

    x = np.array(x).reshape(-1, affect_length, 1)
    t = np.array(t).reshape(-1, 1)


    ipdb.set_trace()

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
    batch_size = 100
    n_batches = x.shape[0] // batch_size + 1
    hist = {'loss': [], 'val_loss': []}
    all_loss = []
    # es = EarlyStopping(patience=10, verbose=1)

    for epoch in range(epochs):
        train_loss = 0.
        val_loss = 0.
        x_, t_ = shuffle(x, t)

        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            loss, _ = train_step(x_[start:end], t_[start:end])
            all_loss.append(loss.item())
            train_loss += loss.item()

        train_loss /= n_batches
        print('epoch: {}, loss: {:.3}'.format(epoch+1,train_loss))

    '''
    4. モデルの評価
    '''
    model.eval()

    # sin波の予測
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
    # plt.ylim([-1.5, 1.5])
    # plt.plot(range(len(y)), sin,color='gray',linestyle='--', linewidth=0.5)
    # plt.plot(range(len(y)), gen,color='black', linewidth=1,marker='o', markersize=1, markerfacecolor='black',markeredgecolor='black')
    # plt.savefig('output.jpg')
    # plt.show()


    ipdb.set_trace()


# lossが10万個取りたい。(batch10 * epochs 1000)
