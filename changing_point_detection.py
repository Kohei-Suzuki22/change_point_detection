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
from sklearn.metrics import mean_squared_error
import time
import statistics
import ipdb
import math
from matplotlib import mathtext
import pylab as plt
mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants


plt.rcParams["font.size"] = 10
# plt.tight_layout()
fig = plt.figure(figsize=(20.0,12.0/0.96))



class RNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # self.l1 = nn.RNN(1, hidden_dim,batch_first=True)
        self.l1 = nn.RNN(1, hidden_dim,batch_first=True,bidirectional=True)

        # 入力層(第一層) →  入力:1,出力:hidden_dim
        # 出力層(第二層) →  入力:hidden_dim,出力:1
        # self.l2 = nn.Linear(hidden_dim, 1)
        self.l2 = nn.Linear(hidden_dim*2, 1)

        nn.init.xavier_normal_(self.l1.weight_ih_l0)
        nn.init.orthogonal_(self.l1.weight_hh_l0)

    # 順伝播
    def forward(self, x):
        h, _ = self.l1(x)       # h.shape: [25,1,2] → [batch_size,affect_length,n_hidden]
        y = self.l2(h[:, -1])   # y.shape: [25,1]   → [予測結果(batch_size分)]
        return y                # 予測結果を出力.

class LSTM(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        # self.l1 = nn.LSTM(1,hidden_dim,batch_first=True)
        self.l1 = nn.LSTM(1,hidden_dim,batch_first=True,bidirectional=True)
        # self.l2 = nn.Linear(hidden_dim,1)
        self.l2 = nn.Linear(hidden_dim*2,1)
        nn.init.xavier_normal_(self.l1.weight_ih_l0)
        nn.init.orthogonal_(self.l1.weight_hh_l0)

    def forward(self,x):
        h, _ = self.l1(x)
        y = self.l2(h[:,-1])
        return y

class GRU(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()

        self.l1 = nn.LSTM(1,hidden_dim,batch_first=True,bidirectional=True)
        # self.l2 = nn.Linear(hidden_dim,1)
        self.l2 = nn.Linear(hidden_dim*2,1)
        nn.init.xavier_normal_(self.l1.weight_ih_l0)
        nn.init.orthogonal_(self.l1.weight_hh_l0)

    def forward(self,x):
        h, _ = self.l1(x)
        y = self.l2(h[:,-1])
        return y

class MLP(nn.Module):
    '''
    多層パーセプトロン
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim,hidden_dim)
        self.a1 = nn.Tanh()
        self.l2 = nn.Linear(hidden_dim,output_dim)

        self.layers = [self.l1, self.a1, self.l2]

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)

        return x

if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    # deviceに実行環境を格納することで同じコードをCPU,GPUどちらも対応.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 実行環境が CPU か GPU かを判別して適用。

    '''
    1. データの準備
    '''
    affect_length = 1

    def func(x,t,a=4):
        return a * x[t] * (1 - x[t])

    def y_init():
        return np.array([0.2])

    def set_y_params(y,before_a=3.7,after_a=4):
        for t in range(500):
            if t < 250:
                y = np.append(y,func(y,t,before_a))
            else:
                y = np.append(y,func(y,t,after_a))
        return y

    def factors_answers_init():
        # factors = []
        factors = np.array([])
        # answers = []
        answers = np.array([])
        return (factors,answers)

    def set_factors_answers(y,factors,answers):
        for i in range(len(y) - affect_length):
            # factors.append(y[i:i+affect_length])
            factors = np.append(factors,y[i:i+affect_length])
            # answers.append(y[i+affect_length])
            answers = np.append(answers,y[i+affect_length])
        # factors = np.array(factors).reshape(-1, affect_length, 1)
        factors = factors.reshape(-1,affect_length,1)
        # answers = np.array(answers).reshape(-1, 1)
        answers = answers.reshape(-1,1)
        return (factors,answers)


    def make_dataset(before_a=3.7,after_a=4):
        y = y_init()
        y = set_y_params(y,before_a,after_a)
        factors,answers = factors_answers_init()
        factors,answers = set_factors_answers(y,factors,answers)
        return (y,factors,answers)


    # 誤差を計算(平均二乗誤差).
    def compute_loss(preds,answers,criterion):
        return criterion(preds, answers)

    # 学習ステップ
    def train_step(factors, answers,model,criterion,optimizer):
        # factors,answersを pytorch用データに変換。
        factors = torch.Tensor(factors).to(device)
        answers = torch.Tensor(answers).to(device)

        model.train()                       # モデルに「学習モードになれ」と伝える。
        preds = model(factors)              # forwardメソッド実行。
        loss = compute_loss(preds,answers,criterion)  # 誤差は平均二乗誤差.
        optimizer.zero_grad()
        loss.backward()                     # 誤差逆伝播.
        optimizer.step()                    # パラメータの更新.
        return loss, preds


    epochs = 1000
    batch_size = 1

    # 学習実行 & 学習損失GET.
    def get_loss(factors,answers,model,criterion,optimizer,picture_name,before_a,after_a):
        n_batches = factors.shape[0] // batch_size      # -> 20 = 500 / 25
        hist = {'loss': []}
        # es = EarlyStopping(patience=10, verbose=1)
        for epoch in range(epochs):
            train_loss = 0.
            loss_per_batch = np.array([])
            preds_per_batch = np.array([])
            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                loss, preds = train_step(factors[start:end], answers[start:end],model,criterion,optimizer)
                train_loss += loss.data
                loss_per_batch = np.append(loss_per_batch,loss.data)
                preds_per_batch = np.append(preds_per_batch,preds.data)

            train_loss /= n_batches
            hist['loss'].append(train_loss)
            print('epoch: {}, loss: {:.3}'.format(epoch+1,train_loss))
            # if es(train_loss):
            #     break
        return (loss_per_batch,preds_per_batch)



    # グラフの作成
    def show_raw_graph(y):
        plt.rc('font', family='serif')
        plt.subplot(3,1,1)
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.plot(range(len(y)), y, linewidth=1,color="blue",label="row_data")
        plt.legend(loc="lower left")

    def show_graph_compare_raw_preds(answers,preds,picture_name):
        # plt.rc('font', family='serif')
        plt.subplot(3,1,3)
        # fig = plt.figure(figsize=(12.0,8.0))
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.plot(range(len(answers)), answers, linewidth=1,color="blue",label="row_data")
        plt.plot(range(len(preds)),preds,linewidth=0.6,color="red",label="pred")
        plt.legend(loc="lower left")
        # fig.savefig("gru_pictures/pictures_neuron2/epochs_1000/batch_1/bidirectional/system_{0}_epochs{1}_range0.01.png".format(picture_name,epochs))
        # return plt
    
    def show_graph_loss_per_batch(loss_per_batch,picture_name,before_a,after_a,epochs):
        # fig = plt.figure(figsize=(12.0,8.0))
        plt.subplot(3,1,2)
        plt.plot(loss_per_batch,color="blue",label="{0}~{1} (epochs={2})".format(before_a,after_a,epochs))
        # plt.xlim(0,21)
        # plt.xticks([0,50,100,150,200,250,300,350,400,450,500])
        # plt.xticks([0,5,10,15,20])
        plt.xlabel("i")
        # plt.yticks([0.001,0.002,0.003,0.004,0.005])
        plt.ylabel("$E_i$")             # iを下付き文字に変換。
        plt.legend(loc="lower right")
        # fig.savefig("gru_pictures/pictures_neuron2/epochs_1000/batch_1/bidirectional/change_point_{0}_epochs{1}_range0.01.png".format(picture_name,epochs))
        # return plt


    def execute_all(picture_name,model,before_a=3.7,after_a=4):
          # 隠れ層2ニューロンのモデル生成. (RNN(ニューロン数). 2: 凸凹幅大きい。 ←→ ニューロン数200: 凸凹幅小さい。)

        criterion = nn.MSELoss(reduction='mean')    # 損失関数: 平均二乗誤差
        optimizer = optimizers.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True)     # 最適化手法


        y,factors,answers = make_dataset(before_a,after_a)
        loss_per_batch,preds_per_batch = get_loss(factors,answers,model,criterion,optimizer,picture_name,before_a,after_a)
        show_raw_graph(y)
        show_graph_compare_raw_preds(answers,preds_per_batch,picture_name)
        show_graph_loss_per_batch(loss_per_batch,picture_name,before_a,after_a,epochs)
        plt.show()


    # model = MLP(1,4,1).to(device)
    # model = RNN(2).to(device)
    model = LSTM(2).to(device)
    # model = GRU(2).to(device)
    execute_all(0,model,3.90,4.0)

    # for i in range(1000):
    #     a_start = round(3.7 +(i / 1000),4)
    #     if (a_start > 4.0):
    #         break
    #     # model = MLP(1,4,1).to(device)
    #     # model = RNN(2).to(device)
    #     # model = LSTM(2).to(device)
    #     model = GRU(2).to(device)
    #     execute_all(i,model,a_start,4.0)

    # for i in range(1000):
    #     plus_range = 0.1
    #     a_start = round(3.7+(i/10)+plus_range,4)
    #     a_end = round(a_start+plus_range,4)
    #     if(a_end>4.0):
    #         break
    #     execute_all(i,a_start,a_end)

    # y = y_init
    # params = [3.7, 3.8, 3.85, 3.9, 3.95, 3.99]
    # for param in params:
    #     y = set_y_params(y_init(),param,4)
    #     plt.subplot(2,3,params.index(param)+1)
    #     plt.plot(y,label="a = {0} → 4.0".format(param))
    #     plt.legend(loc="lower right")

    # y = set_y_params(y_init(),3.7,4)
    # # plt.subplot(2,3,params.index(param)+1)
    # plt.plot(y,label="a = {0} → 4.0".format(3.7))
    # plt.legend(loc="lower right")

