# 漸化式 x(t+1) = 4x(t) (1-x(t))の時系列解析。
# ノイズなし。

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential

from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import time
import statistics
import ipdb

np.random.seed(0) #乱数の固定

# 波形の生成

def func(x,t,a=4):
  return a * x[t] * (1 - x[t])

y = np.array([0.2])

for t in range(499):
  if t < 300:
    y = np.append(y,func(y,t))
  else:
    y = np.append(y,func(y,t,2))

# for t in range(499):
#   y = np.append(y,func(y,t))

t = np.arange(500)

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


n_in = 1
n_middle = 1
n_out = 1
n_hidden_unit = 100
lr = 0.0001

# 二乗平均平方根誤差
def rmse(y_true,y_pred):
  return round(np.sqrt(mean_squared_error(y_true,y_pred)),3)

def rnn_test(learning_rate=0.0001,affect_length=1,activation='hard_sigmoid',epochs=3,loss_func='mean_squared_error',num_neurons=1, n_hidden=1,batch_size=1,patience=30,validation_split=0.05):


    model = Sequential()
    model.add(SimpleRNN(n_hidden, batch_input_shape=(None, affect_length, num_neurons), return_sequences=False))
    # model.add(LSTM(n_hidden, batch_input_shape=(None, affect_length, num_neurons), return_sequences=False))
    model.add(Dense(num_neurons))
    model.add(Activation(activation))
    optimizer = Adam(lr = lr)
    model.compile(loss=loss_func, optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=patience)
    ipdb.set_trace()
    hist = model.fit(factors, answers, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[])
    # hist = model.fit(factors, answers,epochs=epochs, validation_split=validation_split, callbacks=[])
    ipdb.set_trace()
    pred = model.predict(factors)
    y_true = y[affect_length:]
    y_pred = pred.reshape(-1)
    # ipdb.set_trace()




    # plt.subplot(1, 2, 1)
    plt.title("learning_rate={}, rmse={}".format(lr,rmse(y_true[-25:],y_pred[-25:])))
    plt.xlim(250, 350)
    plt.plot(t, y, color='red', label='raw_data')
    plt.plot(t[affect_length:], y_pred, color='blue', label='predicted')
    plt.legend(loc='upper right', ncol=2)

    plt.show()

    # ipdb.set_trace()

    # plt.subplot(1, 2, 2)
    plt.title("learning_rate={}, batch_size={}".format(lr,batch_size))
    loss = hist.history["loss"]
    plt.plot(range(len(loss)),loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

    # ipdb.set_trace()

    # plt.subplot(1, 3, 3)
    plt.title("learning_rate={}, batch_size={}".format(lr,batch_size))
    val_loss = hist.history["val_loss"]
    plt.plot(range(len(val_loss)),val_loss)
    plt.xlabel("eochs")
    plt.ylabel("val_loss")
    plt.show()


rnn_test(batch_size=1)







'''
※ 学習中の loss, val_lossの違い
loss -> 学習用データを与えた際の損失値
        小さいほど学習出来たことを表す。
val_loss -> 検証用データを与えた際の損失値
            小さいほど正しい結果(検証データに近い値)を出せた。
'''