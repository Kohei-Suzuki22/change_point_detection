# 漸化式 x(t+1) = 4x(t) (1-x(t))の時系列解析。
# ノイズなし。

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



np.random.seed(0) #乱数の固定

# 波形の生成

def func(x,t,a=4):
  return a * x[t] * (1 - x[t])

y = np.array([0.2])

for t in range(499):
  if t < 300:
    y = np.append(y,func(y,t))
  else:
    y = np.append(y,func(y,t,3))

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
# print(factors.shape)
# print(answers.shape)
# print(factors.shape)      #: (268,32,1)
# print(answers.shape)      #: (268,)

n_in = 1
n_middle = 1
n_out = 1
n_hidden_unit = 100
lr = 0.0001



# 二乗平均平方根誤差
def rmse(y_true,y_pred):
  return round(np.sqrt(mean_squared_error(y_true,y_pred)),3)

# モデル構築

def normal_graph():
  model = Sequential()

  model.add(SimpleRNN(n_hidden_unit, batch_input_shape=(None, affect_length, n_middle), return_sequences=False))

  # 差再帰層から出力層までの定義(Dense,Activation)

  # model.add(Dense(n_middle)) #Dense == 全結合モデル。   n_hidden: 隠れ層の数
  # model.add(Activation('linear'))
  #
  model.add(Dense(n_middle))
  model.add(Activation('tanh'))
  # 隠れ層2層目
  model.add(Dense(n_middle))
  model.add(Activation('linear'))
  # model.add(Activation('hard_sigmoid'))

  optimizer = Adam(lr = lr)
  # Adam: 最適化手法の一つ。 デファクトスタンダート。


  # モデルのコンパイル

  model.compile(loss="mean_absolute_error", optimizer=optimizer)


  # 学習の進み具合に応じて、エポックを実行るか打ち切るかを判断。
  early_stopping = EarlyStopping(monitor='val_loss',  mode='auto', patience=300)
  ## monitor: 監視する値
  ## mode: 訓練を終了するタイミング{auto,min,max}
      # → min: 監視する値の減少が停止した際に訓練終了
      # → max: 監視する値の増加が停止した際に訓練終了
      # → auto: minかmaxか、自動的に推測
  ## patience: 指定したエポック数の間に改善がないと訓練終了


  print(hi)
  ipdb.set_trace()



  # 学習

  ## バッチサイズ: 128
  ## エポック数: 100


  ## .fit: 学習を実行
  ## 学習データ: factors
  ## 正解データ(教師データ): answers
  ## validation_split: テストデータとしての割合を指定。
    ## callbacks=[]: 訓練中に適応される関数たちを登録。

  # model.fit(factors,answers,batch_size=128,
  # epochs=2000,validation_split=0.05, callbacks= [early_stopping])
  # epochs=2000,validation_split=0.05, callbacks=[])


  # 予測
  pred = model.predict(factors)
  # print(pred)


# グラフ表示


  y_pred = pred.reshape(-1)
  y_true = y[affect_length:]

  print("---学習データ+検証データ---")
  print(rmse(y_true,y_pred))

  print("---検証データ---")
  print(rmse(y_true[-25:],y_pred[-25:]))

  # ipdb.set_trace()
  # plt.xlim(400, 550)
  plt.plot(t, y, color='blue', label='raw_data')
  plt.show()
  plt.plot(t[affect_length:], pred, color='red', label='pred')
  plt.xlabel('t')
  plt.legend(loc='lower left')  # 図のラベルの位置を指定。

# plt.tight_layout()            #グラフの重なりを解消。
  plt.show()


# normal_graph()


# learning_rate
# affect_length
# activation
# loss(損失関数)
# n_hidden
# epochs
# batch_size
# num_neurons


def rnn_test_per_learning_rate(learning_rate=[0.01],affect_length=2,activation='hard_sigmoid',epochs=2000,loss_func='mean_squared_error',num_neurons=1, n_hidden=20,batch_size=32,patience=300,validation_split=0.05,width=2, height=2,):
  plt.figure(figsize=(20, 20))
  for i, lr in enumerate(learning_rate):
    (factors, answers) = make_dataset(y, affect_length)
    factors = np.array(factors).reshape(-1, affect_length, 1)

    model = Sequential()
    model.add(SimpleRNN(n_hidden, batch_input_shape=(None, affect_length, num_neurons), return_sequences=False))
    model.add(Dense(num_neurons))
    model.add(Activation(activation))
    optimizer = Adam(lr = lr)
    model.compile(loss=loss_func, optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=patience)
    model.fit(factors, answers, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[])

    pred = model.predict(factors)
    y_true = y[affect_length:]
    y_pred = pred.reshape(-1)

    # ipdb.set_trace()
    plt.subplot(width, height, i+1)
    plt.title("learning_rate={}, rmse={}".format(lr,rmse(y_true[-25:],y_pred[-25:])))
    plt.xlim(250, 350)
    plt.plot(t, y, color='red', label='raw_data')
    plt.plot(t[affect_length:], y_pred, color='blue', label='predicted')
    plt.legend(loc='upper right', ncol=2)

  plt.show()

learning_rate=[0.1]
rnn_test_per_learning_rate(learning_rate, width=1, height=1)

# 学習率は 0.01が最善かも。



def rnn_test_per_affect_length(learning_rate=0.01,affect_length=[2],activation='hard_sigmoid',epochs=2000,loss_func='mean_squared_error',num_neurons=1, n_hidden=20,batch_size=32,patience=300,validation_split=0.05,width=2, height=2, ):
  plt.figure(figsize=(20, 20))
  for i, al in enumerate(affect_length):
    (factors, answers) = make_dataset(y, al)
    factors = np.array(factors).reshape(-1, al, 1)

    model = Sequential()
    model.add(SimpleRNN(n_hidden, batch_input_shape=(None, al, num_neurons), return_sequences=False))
    model.add(Dense(num_neurons))
    model.add(Activation(activation))
    optimizer = Adam(lr = learning_rate)
    model.compile(loss=loss_func, optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=patience)
    model.fit(factors, answers, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[])

    pred = model.predict(factors)
    y_true = y[al:]
    y_pred = pred.reshape(-1)

    plt.subplot(width, height, i+1)
    plt.title("affect_length={}, rmse={}".format(al,rmse(y_true[-25:],y_pred[-25:])))
    plt.xlim(450, 510)
    plt.plot(t, y, color='red', label='raw_data')
    plt.plot(t[al:], y_pred, color='blue', label='predicted')
    plt.legend(loc='upper right', ncol=2)

  plt.show()

Affect_Length=[1, 2, 4, 8, 16, 32, 64, 128]
# rnn_test_per_affect_length(affect_length=Affect_Length, width=2, height=4)

# affect_length は 2ぐらいが良かった。



def rnn_test_per_activation(learning_rate=0.01,affect_length=2,activation=["hard_sigmoid"],epochs=2000,loss_func='mean_squared_error',num_neurons=1, n_hidden=20,batch_size=32,patience=300,validation_split=0.05,width=2, height=2, ):
  # plt.figure(figsize=(20, 20))
  for i, ac in enumerate(activation):
    # print(ac)
    # print(type(ac))
    (factors, answers) = make_dataset(y, affect_length)
    factors = np.array(factors).reshape(-1, affect_length, 1)

    model = Sequential()
    model.add(SimpleRNN(n_hidden, batch_input_shape=(None, affect_length, num_neurons), return_sequences=False))
    model.add(Dense(num_neurons))
    model.add(Activation(ac))
    optimizer = Adam(lr = learning_rate)
    model.compile(loss=loss_func, optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=patience)
    model.fit(factors, answers, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[])
    pred = model.predict(factors)

    y_true = y[affect_length:]
    y_pred = pred.reshape(-1)
    plt.subplot(width, height, i+1)
    plt.title("{}, rmse={}".format(ac,rmse(y_true[-25:],y_pred[-25:])))
    plt.xlim(450, 510)
    plt.plot(t, y, color='red', label='raw_data')
    plt.plot(t[affect_length:], y_pred, color='blue', label='predicted')
    plt.legend(loc='upper right', ncol=2)

  plt.show()

activation=['linear','elu','selu','sigmoid','hard_sigmoid','softmax','softplus','softsign','tanh','relu']
# rnn_test_per_activation(activation=activation,width=2, height=5)


# 活性化関数は sigmoidがベスト。



def rnn_test_per_loss_func(learning_rate=0.01,affect_length=2,activation="hard_sigmoid",epochs=2000,loss_func=['mean_squared_error'],num_neurons=1, n_hidden=20,batch_size=32,patience=300,validation_split=0.05,width=2, height=2, ):
  # plt.figure(figsize=(20, 20))
  for i, lf in enumerate(loss_func):
    # print(ac)
    # print(type(ac))
    (factors, answers) = make_dataset(y, affect_length)
    factors = np.array(factors).reshape(-1, affect_length, 1)

    model = Sequential()
    model.add(SimpleRNN(n_hidden, batch_input_shape=(None, affect_length, num_neurons), return_sequences=False))
    model.add(Dense(num_neurons))
    model.add(Activation(activation))
    optimizer = Adam(lr = learning_rate)
    model.compile(loss=lf, optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=patience)
    model.fit(factors, answers, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[])
    pred = model.predict(factors)

    y_true = y[affect_length:]
    y_pred = pred.reshape(-1)
    plt.subplot(width, height, i+1)
    plt.title("{}, rmse={}".format(lf,rmse(y_true[-25:],y_pred[-25:])))
    plt.xlim(450, 510)
    plt.plot(t, y, color='red', label='raw_data')
    plt.plot(t[affect_length:], y_pred, color='blue', label='predicted')
    plt.legend(loc='upper right', ncol=2)

  plt.show()

loss_func = ['mean_squared_error','mean_absolute_error','mean_squared_logarithmic_error']
# rnn_test_per_loss_func(loss_func=loss_func,width=1, height=3)

# 損失関数: mean_squared_error が良かった。


def rnn_test_per_epochs(learning_rate=0.01,affect_length=2,activation="hard_sigmoid",epochs=[2000],loss_func='mean_squared_error',num_neurons=1, n_hidden=20,batch_size=32,patience=300,validation_split=0.05,width=2, height=2, ):
  # plt.figure(figsize=(20, 20))
  for i, ep in enumerate(epochs):
    # print(ac)
    # print(type(ac))
    (factors, answers) = make_dataset(y, affect_length)
    factors = np.array(factors).reshape(-1, affect_length, 1)

    model = Sequential()
    model.add(SimpleRNN(n_hidden, batch_input_shape=(None, affect_length, num_neurons), return_sequences=False))
    model.add(Dense(num_neurons))
    model.add(Activation(activation))
    optimizer = Adam(lr = learning_rate)
    model.compile(loss=loss_func, optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=patience)
    model.fit(factors, answers, batch_size=batch_size, epochs=ep, validation_split=validation_split, callbacks=[])
    pred = model.predict(factors)

    y_true = y[affect_length:]
    y_pred = pred.reshape(-1)
    plt.subplot(width, height, i+1)
    plt.title("epochs:{}, rmse:{}".format(ep,rmse(y_true[-25:],y_pred[-25:])))
    plt.xlim(450, 510)
    plt.plot(t, y, color='red', label='raw_data')
    plt.plot(t[affect_length:], y_pred, color='blue', label='predicted')
    plt.legend(loc='upper right', ncol=2)

  plt.show()

epochs = [100,200,500,1000,2000,5000]
# rnn_test_per_epochs(epochs=epochs,width=2, height=3)

# epoch数は多ければ多いほど、精度が良くなるが時間がかかる。


def rnn_test_per_batch(learning_rate=0.01,affect_length=2,activation="hard_sigmoid",epochs=2000,loss_func='mean_squared_error',num_neurons=1, n_hidden=20,batch_size=[32],patience=300,validation_split=0.05,width=2, height=2, ):
  # plt.figure(figsize=(20, 20))
  for i, batch in enumerate(batch_size):
    (factors, answers) = make_dataset(y, affect_length)
    factors = np.array(factors).reshape(-1, affect_length, 1)

    model = Sequential()
    model.add(SimpleRNN(n_hidden, batch_input_shape=(None, affect_length, num_neurons), return_sequences=False))
    model.add(Dense(num_neurons))
    model.add(Activation(activation))
    optimizer = Adam(lr = learning_rate)
    model.compile(loss=loss_func, optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=patience)
    model.fit(factors, answers, batch_size=batch, epochs=epochs, validation_split=validation_split, callbacks=[])
    pred = model.predict(factors)

    y_true = y[affect_length:]
    y_pred = pred.reshape(-1)
    plt.subplot(width, height, i+1)
    plt.title("batch_size:{}, rmse:{}".format(batch,rmse(y_true[-25:],y_pred[-25:])))
    plt.xlim(450, 510)
    plt.plot(t, y, color='red', label='raw_data')
    plt.plot(t[affect_length:], y_pred, color='blue', label='predicted')
    plt.legend(loc='upper right', ncol=2)

  plt.show()

batch = [1,2,4]
# rnn_test_per_batch(batch_size=batch,width=1, height=3)


# 32が良好。バッチが小さいほど精度良い気がする。


def rnn_test_per_n_hidden(learning_rate=0.01,affect_length=2,activation="hard_sigmoid",epochs=2000,loss_func='mean_squared_error',num_neurons=1, n_hidden=20,batch_size=32,patience=300,validation_split=0.05,width=2, height=2, ):
  # plt.figure(figsize=(20, 20))
  for i, n in enumerate(n_hidden):
    (factors, answers) = make_dataset(y, affect_length)
    factors = np.array(factors).reshape(-1, affect_length, 1)

    model = Sequential()
    model.add(SimpleRNN(n, batch_input_shape=(None, affect_length, num_neurons), return_sequences=False))
    model.add(Dense(num_neurons))
    model.add(Activation(activation))
    optimizer = Adam(lr = learning_rate)
    model.compile(loss=loss_func, optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=patience)
    model.fit(factors, answers, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[])
    pred = model.predict(factors)

    y_true = y[affect_length:]
    y_pred = pred.reshape(-1)
    plt.subplot(width, height, i+1)
    plt.title("n_hidden:{}, rmse:{}".format(n,rmse(y_true[-25:],y_pred[-25:])))
    plt.xlim(450, 510)
    plt.plot(t, y, color='red', label='raw_data')
    plt.plot(t[affect_length:], y_pred, color='blue', label='predicted')
    plt.legend(loc='upper right', ncol=2)

  plt.show()

n_hidden=[1,20,100,200,500,1000]
# rnn_test_per_n_hidden(n_hidden=n_hidden,width=2, height=3)




def rnn_test(learning_rate=0.01,affect_length=2,activation="hard_sigmoid",epochs=2000,loss_func='mean_squared_error',num_neurons=1, n_hidden=20,batch_size=32,patience=300,validation_split=0.05):
  (factors, answers) = make_dataset(y, affect_length)
  factors = np.array(factors).reshape(-1, affect_length, 1)
  model = Sequential()
  model.add(SimpleRNN(n_hidden, batch_input_shape=(None, affect_length, num_neurons), return_sequences=False))
  model.add(Dense(num_neurons))
  model.add(Activation(activation))
  optimizer = Adam(lr = learning_rate)
  model.compile(loss=loss_func, optimizer=optimizer)
  early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=patience)
  model.fit(factors, answers, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[])
  pred = model.predict(factors)
  y_true = y[affect_length:]
  y_pred = pred.reshape(-1)
  plt.title("rmse:{}".format(rmse(y_true[-25:],y_pred[-25:])))
  plt.xlim(0, 100)
  plt.plot(t, y, color='red', label='raw_data')
  plt.plot(t[affect_length:], y_pred, color='blue', label='predicted')
  plt.legend(loc='upper right', ncol=2)
  plt.show()

# rnn_test()



# 50回試行して、RMSEの平均・標準偏差を求める。

def rnn_test(learning_rate=0.01,affect_length=2,activation='hard_sigmoid',epochs=2000,loss_func='mean_squared_error',num_neurons=1, n_hidden=20,batch_size=32,patience=300,validation_split=0.05,loop_count=50):
  time_to_learn = np.zeros(loop_count)
  all_rmse = np.zeros(loop_count)


  for i in range(loop_count):
    (factors, answers) = make_dataset(y, affect_length)
    factors = np.array(factors).reshape(-1, affect_length, 1)
    model = Sequential()
    model.add(SimpleRNN(n_hidden, batch_input_shape=(None, affect_length, num_neurons), return_sequences=False))
    model.add(Dense(num_neurons))
    model.add(Activation(activation))
    optimizer = Adam(lr = learning_rate)
    model.compile(loss=loss_func, optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=patience)
    start = time.time()
    model.fit(factors, answers, batch_size=batch_size, epochs=epochs, validation_split=validation_split,callbacks= [])
    elapsed_time = time.time() - start
    time_to_learn[i] = elapsed_time
    pred = model.predict(factors)
    y_true = y[affect_length:]
    y_pred = pred.reshape(-1)
    all_rmse[i] = rmse(y_true[-25:],y_pred[-25:])

  return time_to_learn, all_rmse

def learning_rate_test(target,data):
  for d in data:
    # コーディングテスト用
    # time_to_learn,all_rmse = rnn_test(learning_rate=d,epochs=2,loop_count=5)
    # 統計収集用
    time_to_learn,all_rmse = rnn_test(learning_rate=d)
    with open("./memo.txt",'a') as f:
      f.write("{}=[{}]:     rmse平均={}, rmse標準偏差={}, 平均実行時間={}\n".format(target,d, round(np.mean(all_rmse),4), round(statistics.pstdev(all_rmse),5), round(np.mean(time_to_learn),4)))


# learning_rate=[0.1,0.01,0.001,0.0001]
learning_rate=[0.1,0.01]

# learning_rate_test("learning_rate",learning_rate)


def affect_length_test(target,data):
  for d in data:
    # コーディングテスト用
    # time_to_learn,all_rmse = rnn_test(affect_length=d,epochs=1,loop_count=50)
    # 統計収集用
    time_to_learn,all_rmse = rnn_test(affect_length=d)
    with open("./memo.txt",'a') as f:
      f.write("{}=[{}]:     rmse平均={}, rmse標準偏差={}, 平均実行時間={}\n".format(target,d, round(np.mean(all_rmse),4), round(statistics.pstdev(all_rmse),5), round(np.mean(time_to_learn),4)))

Affect_Length=[1, 2, 4, 8, 16, 32, 64, 128]
# affect_length_test("affect_length",Affect_Length)

def activation_test(target,data):
  for d in data:
    # コーディングテスト用
    # time_to_learn,all_rmse = rnn_test(activation=d,epochs=1,loop_count=10)
    # 統計収集用
    time_to_learn,all_rmse = rnn_test(activation=d)
    with open("./memo.txt",'a') as f:
      f.write("{}=[{}]:     rmse平均={}, rmse標準偏差={}, 平均実行時間={}\n".format(target,d, round(np.mean(all_rmse),4), round(statistics.pstdev(all_rmse),5), round(np.mean(time_to_learn),4)))


activation=['linear','elu','selu','sigmoid','hard_sigmoid','softmax','softplus','softsign','tanh','relu']
# activation_test("activation",activation)


def epochs_test(target,data):
  for d in data:
    # コーディングテスト用
    # time_to_learn,all_rmse = rnn_test(epochs=d,loop_count=10)
    # 統計収集用
    time_to_learn,all_rmse = rnn_test(epochs=d)
    with open("./memo.txt",'a') as f:
      f.write("{}=[{}]:     rmse平均={}, rmse標準偏差={}, 平均実行時間={}\n".format(target,d, round(np.mean(all_rmse),4), round(statistics.pstdev(all_rmse),5), round(np.mean(time_to_learn),4)))

epochs = [100,200,500,1000,2000,5000]
# epochs_test("epochs",epochs)


def loss_func_test(target,data):
  for d in data:
    # コーディングテスト用
    # time_to_learn,all_rmse = rnn_test(epochs=2,loss_func=d,loop_count=10)
    # 統計収集用
    time_to_learn,all_rmse = rnn_test(loss_func=d)
    with open("./memo.txt",'a') as f:
      f.write("{}=[{}]:     rmse平均={}, rmse標準偏差={}, 平均実行時間={}\n".format(target,d, round(np.mean(all_rmse),4), round(statistics.pstdev(all_rmse),5), round(np.mean(time_to_learn),4)))

loss_func = ['mean_squared_error','mean_absolute_error','mean_squared_logarithmic_error']
# loss_func_test("loss_func",loss_func)



def n_hidden_test(target,data):
  for d in data:
    # コーディングテスト用
    # time_to_learn,all_rmse = rnn_test(epochs=1,n_hidden=d,loop_count=10)
    # 統計収集用
    time_to_learn,all_rmse = rnn_test(n_hidden=d)
    with open("./memo.txt",'a') as f:
      f.write("{}=[{}]:     rmse平均={}, rmse標準偏差={}, 平均実行時間={}\n".format(target,d, round(np.mean(all_rmse),4), round(statistics.pstdev(all_rmse),5), round(np.mean(time_to_learn),4)))

n_hidden=[1,20,100,200,500,1000]
# n_hidden_test("n_hidden",n_hidden)


def batch_size_test(target,data):
  for d in data:
    # コーディングテスト用
    # time_to_learn,all_rmse = rnn_test(epochs=1,batch_size=d,loop_count=10)
    # 統計収集用
    time_to_learn,all_rmse = rnn_test(batch_size=d)
    with open("./memo.txt",'a') as f:
      f.write("{}=[{}]:     rmse平均={}, rmse標準偏差={}, 平均実行時間={}\n".format(target,d, round(np.mean(all_rmse),4), round(statistics.pstdev(all_rmse),5), round(np.mean(time_to_learn),4)))

batch_size = [1,2,4,8,16,32,64,128,256]
# batch_size_test("batch_size",batch_size)
