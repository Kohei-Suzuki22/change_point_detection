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


plt.rcParams["font.size"] = 20
# plt.tight_layout()
plt.figure(figsize=(12.0,8.0))



class RNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 入力層(第一層) →  入力:1,出力:hidden_dim
        self.l1 = nn.RNN(1, hidden_dim,nonlinearity='tanh',batch_first=True)
        # 出力層(第二層) →  入力:hidden_dim,出力:1
        self.l2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.l1.weight_ih_l0)
        nn.init.orthogonal_(self.l1.weight_hh_l0)

    # 順伝播
    def forward(self, x):
        h, _ = self.l1(x)       # h.shape: [25,1,2] → [batch_size,affect_length,n_hidden]
        y = self.l2(h[:, -1])   # y.shape: [25,1]   → [予測結果(batch_size分)]
        return y                # 予測結果を出力.

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
        factors = []
        answers = []
        return (factors,answers)

    def set_factors_answers(y,factors,answers):
        for i in range(len(y) - affect_length):
            factors.append(y[i:i+affect_length])
            answers.append(y[i+affect_length])
        factors = np.array(factors).reshape(-1, affect_length, 1)
        answers = np.array(answers).reshape(-1, 1)
        return (factors,answers)


    def make_dataset(before_a=3.7,after_a=4):
        y = y_init()
        y = set_y_params(y,before_a,after_a)
        factors,answers = factors_answers_init()
        factors,answers = set_factors_answers(y,factors,answers)
        return (y,factors,answers)

    # y,factors,answers = make_dataset(3.7,4)





    '''
    2. モデルの構築
    '''
    # 隠れ層2ニューロンのモデル生成. (RNN(ニューロン数). 2: 凸凹幅大きい。 ←→ ニューロン数200: 凸凹幅小さい。)
    # model = RNN(50).to(device)

    '''
    3. モデルの学習
    '''
    # criterion = nn.MSELoss(reduction='mean')    # 損失関数: 平均二乗誤差
    # optimizer = optimizers.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True)     # 最適化手法

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
    batch_size = 25

    # 学習実行 & 学習損失GET.
    def get_loss(factors,answers,model,criterion,optimizer):
        n_batches = factors.shape[0] // batch_size      # -> 20 = 500 / 25
        hist = {'loss': []}
        all_loss = []
        loss_per_epochs = []
        # es = EarlyStopping(patience=10, verbose=1)
        for epoch in range(epochs):
            train_loss = 0.
            loss_per_batch = []
            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                loss, _ = train_step(factors[start:end], answers[start:end],model,criterion,optimizer)
                all_loss.append(loss.item())
                train_loss += loss.item()
                loss_per_batch.append(loss.item())
            # print(epochs)
            if (epoch == (epochs-1)):
                plt.plot(loss_per_batch,color="blue",label="Ei")
                plt.xlim(0,21)
                plt.xticks([0,5,10,15,20])
                plt.xlabel("i")
                # plt.yticks([0.001,0.002,0.003,0.004,0.005])
                plt.ylabel("$E_i$")             # iを下付き文字に変換。
                plt.show()


            train_loss /= n_batches
            loss_per_epochs.append(train_loss)
            hist['loss'].append(train_loss)
            print('epoch: {}, loss: {:.3}'.format(epoch+1,train_loss))
            # if es(train_loss):
            #     break
        return (loss_per_epochs,all_loss)

    # loss_per_epochs,all_loss = get_loss(factors,answers)

    '''
    4. モデルの評価
    '''
    def model_eval(y,factors,model):
        model.eval()        # モデルに「評価モードになれ」と伝える。
        predicted = [None for i in range(affect_length)]        # 予測結果を入れる箱。
        z = factors[:1]
        # ipdb.set_trace()
        for i in range(len(y) - affect_length):
            z_ = torch.Tensor(z[-1:]).to(device)
            preds = model(z_).data.cpu().numpy()
            z = np.append(z, preds)[1:]
            z = z.reshape(-1, affect_length, 1)
            predicted.append(preds[0, 0])

        return predicted

    # predicted = model_eval(y,factors)


    # ipdb.set_trace()

    # グラフの作成
    def show_graph(y,predicted,loss_per_epochs,all_loss):
        plt.rc('font', family='serif')
        # plt.xlim(300,400)
        fig = plt.figure()
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.plot(range(len(y)), y, linewidth=1,color="blue",label="row_data")
        plt.plot(range(len(y)), predicted, linewidth=0.7,color="red",label="pred")
        plt.legend(loc="lower left")
        plt.show()
        # ipdb.set_trace()

        # plt.ylabel("y")
        # plt.xlabel("epochs")
        # plt.xlim(0,1000)

        # plt.plot(loss_per_epochs,color="blue",label="loss_per_epoch")
        # plt.show()
        plt.xlabel("all_loss(per_batch)")
        plt.ylabel("y")

        # plt.xlim(10000,11000)
        # plt.xlim(0,1000)
        plt.plot(all_loss,color="blue",label="loss_per_batch(all_loss)")
        plt.show()
        # ipdb.set_trace()

    # show_graph(y,predicted,loss_per_epochs,all_loss)



    def execute_all(before_a=3.7,after_a=4):
          # 隠れ層2ニューロンのモデル生成. (RNN(ニューロン数). 2: 凸凹幅大きい。 ←→ ニューロン数200: 凸凹幅小さい。)
        model = RNN(50).to(device)
        criterion = nn.MSELoss(reduction='mean')    # 損失関数: 平均二乗誤差
        optimizer = optimizers.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True)     # 最適化手法


        y,factors,answers = make_dataset(before_a,after_a)
        loss_per_epochs,all_loss = get_loss(factors,answers,model,criterion,optimizer)
        predicted = model_eval(y,factors,model)
        show_graph(y,predicted,loss_per_epochs,all_loss)

    execute_all()
    execute_all(3.75,3.95)
    execute_all(3.8,3.9)
    execute_all(3.85,3.9)
    execute_all()