# 漸化式 x(t+1) = 4x(t) (1-x(t))の時系列解析。
# ノイズなし。

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


np.random.seed(0) #乱数の固定

# 波形の生成

def func(x,t,a=4):
  return a * x[t] * (1 - x[t])

y = np.array([0.2])

for t in range(299):
  y = np.append(y,func(y,t))

t = np.arange(300)


affect_length = 32


def make_dataset(y, affect_length):
  factors = np.array([[]])
  answers = np.array([])
  print(y[0:0+affect_length].shape)
  for i in range(len(y) - affect_length):
    # print(y[i:i+affect_length])
    factors = np.append(factors,y[i:i+affect_length])
    answers = np.append(answers,y[i+affect_length])

  return(factors, answers)

(factors, answers) = make_dataset(y,affect_length)

factors = factors.reshape(-1,affect_length,1)
# make_dataset(y,affect_length)
# print(factors.shape)      #: (268,32,1)
# print(answers.shape)      #: (268,)


# モデルの定義

## 入力層: 1
## 隠れ層: 1
## 隠れ層ユニット数: 200
## 出力層: 1

## 活性化関数: linear
## 誤差関数(損失関数): 平均二乗誤差
## 学習法: 勾配降下法
## 学習率(lr): 0.001

n_in = 1
n_middle = 1
n_out = 1
n_hidden_unit = 200
lr = 0.0001


# モデル構築
model = Sequential()

model.add(SimpleRNN(n_hidden_unit, batch_input_shape=(None, affect_length, n_middle), return_sequences=False))

# 差再帰層から出力層までの定義(Dense,Activation)
model.add(Dense(n_middle)) #Dense == 全結合モデル。 n_hidden: 隠れ層の数
model.add(Activation('linear'))

optimizer = Adam(lr = lr)
# Adam: 最適化手法の一つ。 デファクトスタンダート。


# モデルのコンパイル

model.compile(loss="mean_squared_error", optimizer=optimizer)


# 学習の進み具合に応じて、エポックを実行るか打ち切るかを判断。
early_stopping = EarlyStopping(monitor='val_loss',mode='auto', patience=20)
## monitor: 監視する値
## mode: 訓練を終了するタイミング{auto,min,max}
    # → min: 監視する値の減少が停止した際に訓練終了
    # → max: 監視する値の増加が停止した際に訓練終了
    # → auto: minかmaxか、自動的に推測
## patience: 指定したエポック数の間に改善がないと訓練終了




# 学習

## バッチサイズ: 128
## エポック数: 100


## .fit: 学習を実行
## 学習データ: factors
## 正解データ(教師データ): answers
## validation_split: テストデータとしての割合を指定。
## callbacks=[]: 訓練中に適応される関数たちを登録。

model.fit(factors,answers,batch_size=128,
epochs=1000,validation_split=0.2, callbacks=[early_stopping])


# 予測
pred = model.predict(factors)
# print(pred)


# グラフ表示


plt.plot(t, y, color='blue', label='raw_data')
plt.plot(t[affect_length:], pred, color='red', label='pred')
plt.xlabel('t')
plt.legend(loc='lower left')  # 図のラベルの位置を指定。

plt.tight_layout()            #グラフの重なりを解消。
plt.show()