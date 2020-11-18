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
import os
mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants


# plt.rcParams["font.size"] = 10
# fig = plt.figure(figsize=(20.0,12.0/0.96))



class RNN(nn.Module):
    def __init__(self, hidden_dim,bidirection=False):
        super().__init__()
        if bidirection == True:
            self.l1 = nn.RNN(1, hidden_dim,batch_first=True,bidirectional=True)

            # self.l2 = nn.Linear(hidden_dim*2,hidden_dim*2)
            # self.a2 = nn.Tanh()

            # self.l3 = nn.Linear(hidden_dim*2,hidden_dim*2)
            # self.a3 = nn.Tanh()

            # self.l4 = nn.Linear(hidden_dim*2,hidden_dim*2)
            # self.a4 = nn.Tanh()

            self.l5 = nn.Linear(hidden_dim*2, 1)
        elif bidirection == False:
            self.l1 = nn.RNN(1, hidden_dim,batch_first=True)
            self.l2 = nn.Linear(hidden_dim, 1)

        # 入力層(第一層) →  入力:1,出力:hidden_dim
        # 出力層(第二層) →  入力:hidden_dim,出力:1

        nn.init.xavier_normal_(self.l1.weight_ih_l0)
        nn.init.orthogonal_(self.l1.weight_hh_l0)

        # self.layers = [self.l1, self.l2, self.a2, self.l3, self.a3, self.l4, self.a4, self.l5]
        self.layers = [self.l1,self.l5]


    # 順伝播
    def forward(self, x):

        for layer in self.layers:
            if isinstance(layer,torch.nn.modules.rnn.RNN):
                x = layer(x)[0]
            else:
                x = layer(x)

        return x

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
        x = self.l2(h[:,-1])
        return x

class GRU(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()

        self.l1 = nn.LSTM(1,hidden_dim,batch_first=True,bidirectional=True)
        # self.l2 = nn.Linear(hidden_dim,1)
        self.l2 = nn.Linear(hidden_dim*2,1)
        nn.init.xavier_normal_(self.l1.weight_ih_l0)        #
        nn.init.orthogonal_(self.l1.weight_hh_l0)           #

    def forward(self,x):
        h, _ = self.l1(x)
        x = self.l2(h[:,-1])
        return x

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

class LogisticFunc():

    def __init__(self,before_a,after_a):
        self.before_a = before_a
        self.after_a = after_a
        self.x = np.array([0.2])

    def logistic_func(self,t,a):
            return a * self.x[t] * (1 - self.x[t])

    def make_time_series(self):
        for t in range(500):
            if t < 250:
                self.x = np.append(self.x,self.logistic_func(t,self.before_a))
            else:
                self.x = np.append(self.x,self.logistic_func(t,self.after_a))
        return self.x

class HenonFunc():

    def __init__(self,b,before_a,after_a):
        self.b = 0.3
        self.before_a = before_a
        self.after_a = after_a
        self.x = np.array([0.1])
        self.y = np.array([0.])

    def henon_func_x(self,t,a):
        return 1 - a * self.x[t] ** 2 + self.y[t]

    def henon_func_y(self,t,a):
        return self.b * self.x[t]

    def make_time_series(self):
        for t in range(500):
            if t < 250:
                self.x = np.append(self.x,self.henon_func_x(t,self.before_a))
                self.y = np.append(self.y,self.henon_func_y(t,self.before_a))
            else:
                self.x = np.append(self.x,self.henon_func_x(t,self.after_a))
                self.y = np.append(self.y,self.henon_func_y(t,self.after_a))
        return self.x



if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    # deviceに実行環境を格納することで同じコードをCPU,GPUどちらも対応.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 実行環境が CPU か GPU かを判別して適用。

    '''
    1. データの準備
    '''
    affect_length = 1


    def factors_answers_init():
        factors = np.array([])
        answers = np.array([])
        return (factors,answers)

    def set_factors_answers(x,factors,answers):
        for i in range(len(x) - affect_length):
            factors = np.append(factors,x[i:i+affect_length])
            answers = np.append(answers,x[i+affect_length])
        factors = factors.reshape(-1,affect_length,1)
        answers = answers.reshape(-1,1)         # .shape: (500,1)
        return (factors,answers)

    def make_dataset(target,before_a,after_a):
        factors,answers = factors_answers_init()
        if target == "logistic":
            func = LogisticFunc(before_a,after_a)
        elif target == "henon":
            func = HenonFunc(0.3,before_a,after_a)
        x = func.make_time_series()
        factors,answers = set_factors_answers(x,factors,answers)
        return (x,factors,answers)


    # 誤差を計算(平均二乗誤差).
    def compute_loss(preds,answers,criterion):
        return criterion(preds, answers)

    # 学習ステップ
    def train_step(factors, answers,model,criterion,optimizer):
        # factors,answersを pytorch用データに変換。
        factors = torch.Tensor(factors).to(device)
        answers = torch.Tensor(answers).to(device)
        model.train()                       # モデルに「学習モードになれ」と伝える。
        preds = model(factors)              # forwardメソッド実行。                     # factors.shape: (1,1,1)を入れる必要がある
        loss = compute_loss(preds,answers,criterion)  # 誤差は平均二乗誤差.
        optimizer.zero_grad()
        loss.backward()                     # 誤差逆伝播.
        optimizer.step()                    # パラメータの更新.
        return loss, preds


    epochs = 1000
    batch_size = 1

    # 学習実行 & 学習損失GET.
    def get_loss(factors,answers,model,criterion,optimizer,before_a,after_a):
        n_batches = factors.shape[0] // batch_size
        hist = {'loss': []}
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
        return (loss_per_batch,preds_per_batch)

    def take_move_mean(target,filter_array,mode):
        mean_range = np.ones(filter_array) / filter_array
        move_mean_x = np.convolve(target,mean_range,mode=mode)[:500]
        return move_mean_x

    # グラフの作成
    def show_raw_graph(x,before_a,after_a):
        plt.subplot(3,1,1)
        # plt.title("raw_data")
        plt.xlabel("t")
        plt.xticks([0,50,100,150,200,250,300,350,400,450,500])
        # plt.xticks([220,230,240,250,260,270,280])
        # plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
        # plt.yticks([0,0.5,1.0])
        plt.ylabel("x(t)")
        plt.plot(range(len(x)), x, linewidth=1.0,color="blue",label="{0}~{1}".format(before_a,after_a))
        # plt.legend(loc="lower left")

    def show_graph_compare_raw_preds(answers,preds):
        plt.subplot(2,1,1)
        # plt.title("compare_raw_pred")
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.xticks([0,50,100,150,200,250,300,350,400,450,500])

        # plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
        # plt.yticks([0,0.5,1.0])
        plt.plot(range(len(answers)), answers, linewidth=1,color="blue",label="row_data")
        plt.plot(range(len(preds)),preds,linewidth=0.6,color="red",label="pred")
        # plt.legend(loc="lower left")

    def show_graph_loss_per_batch(loss_per_batch,before_a,after_a,epochs):
        # plt.subplot(3,1,2)
        plt.subplot(2,1,2)
        # title = "loss_flow"
        # plt.title(title)
        plt.plot(range(len(loss_per_batch)),loss_per_batch,color="blue",label="{0}~{1} (epochs={2})".format(before_a,after_a,epochs))
        plt.xlabel("t")
        plt.xticks([0,50,100,150,200,250,300,350,400,450,500])
        plt.ylabel("E(t)")
        # plt.yticks([0.000,0.0005,0.001])
        # plt.legend(loc="lower right")
        interval = 10
        end = loss_per_batch.shape[0]
        start = end // interval


    def show_graph_move_mean_loss_per_batch(loss_per_batch,mean_range,before_a,after_a,epochs):
        plt.subplot(3,1,3)
        move_mean_value = take_move_mean(loss_per_batch,mean_range,"full")
        plt.plot(range(len(move_mean_value)),move_mean_value,color="blue",label="{0}~{1} (epochs={2})".format(before_a,after_a,epochs))
        plt.xlabel("t")
        plt.xticks([0,50,100,150,200,250,300,350,400,450,500])
        plt.ylabel("E(t)")
        # plt.yticks([0.000,0.0005,0.001])


    def execute_all(target,neuron_num,model,before_a,after_a,learning_rate=0.001):
        # 隠れ層2ニューロンのモデル生成. (RNN(ニューロン数). 2: 凸凹幅大きい。 ←→ ニューロン数200: 凸凹幅小さい。)

        criterion = nn.MSELoss(reduction='mean')    # 損失関数: 平均二乗誤差
        optimizer = optimizers.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), amsgrad=True)     # 最適化手法


        x,factors,answers = make_dataset(target,before_a,after_a)

        loss_per_batch,preds_per_batch = get_loss(factors,answers,model,criterion,optimizer,before_a,after_a)

        # plt.rcParams["font.size"] = 37
        plt.rcParams["font.size"] = 24
        # fig = plt.figure(figsize=(20.0,12.0/0.96))

        # show_raw_graph(x,before_a,after_a)
        show_graph_loss_per_batch(loss_per_batch,before_a,after_a,epochs)
        show_graph_compare_raw_preds(answers,preds_per_batch)
        mean_range = 10
        # show_graph_move_mean_loss_per_batch(loss_per_batch,mean_range,before_a,after_a,epochs)

        plt.tight_layout()

        dir_name = "henon_map/RNN/learning_rate{}/neuron{}".format(learning_rate,neuron_num)
        # dir_name = "logistic_map/RNN/learning_rate{}/neuron{}".format(learning_rate,neuron_num)
        os.makedirs(dir_name,exist_ok=True)
        fig.savefig("{}/prediction_accuracy{}to{}_{}epochs.png".format(dir_name,before_a,after_a,epochs))

        # plt.show()

    # params = [3.95,3.99,3.9,3.85,3.8,3.75,3.7]
    # params = [3.7]
    params = [1.00,1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40]
    # params = [1.00]
    for param in params:
        # plt.rcParams["font.size"] = 5
        # neuron_nums = [2,4,8,16]
        neuron_nums = [1]
        for neuron_num in neuron_nums:
            model = RNN(neuron_num,bidirection=True).to(device)
        # execute_all("logistic",0,2,model2,param,4.0)
            fig = plt.figure(figsize=(16.0,8.0/0.96))
            execute_all("henon",neuron_num,model,param,1.4,learning_rate=0.001)
            # execute_all("logistic",neuron_num,model,param,4.0,learning_rate=0.001)

    # for i in range(1000):
    #     plt.rcParams["font.size"] = 10
    #     fig = plt.figure(figsize=(20.0,12.0/0.96))
    #     a_start = round(3.7 +(i / 1000),4)
    #     if (a_start > 4.0):
    #         break
    #     # model = MLP(1,4,1).to(device)
    #     model = RNN(2,bidirection=True).to(device)
    #     # model = RNN(2,bidirection=True).to(device)
    #     # model = LSTM(2).to(device)
    #     # model = GRU(2).to(device)
    #     execute_all(i,model,a_start,4.0)

    # for i in range(1000):
    #     plus_range = 0.1


    #     a_start = round(3.7+(i/10)+plus_range,4)
    #     a_end = round(a_start+plus_range,4)
    #     if(a_end>4.0):
    #         break
    #     execute_all(i,a_start,a_end)



