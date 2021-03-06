# 導入函式庫
from preprocess_v5 import *
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from keras.models import load_model
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import json
import time
import pickle

NAME = f'speechRecog_{int(time.time())}'       # v3 ADD
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
LOG_DIR = f'{int(time.time())}' 

# Save data to array file first
# save_data_to_array(path=DATA_PATH, max_pad_len=11)

# 載入 data 資料夾的訓練資料，並自動分為『訓練組』及『測試組』
X_train, X_test, y_train, y_test = get_train_test()
X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)

# 類別變數轉為one-hot encoding
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
print("X_train.shape=", X_train.shape)

def build_model():  # random search passes this hyperparameter() object 
    
    #建立簡單的線性執行的模型
    model = Sequential()
    # 建立卷積層，filter=32,即 output size, Kernal Size: 2x2, activation function 採用 relu
    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
    # 建立池化層，池化大小=2x2，取最大值
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(192,(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(224,(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(2,2)))
    model.add(Activation('relu'))

    # Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
    model.add(Dropout(0.25))
    # Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
    model.add(Flatten())
    # 全連接層: 128個output
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    # Add output layer
    model.add(Dense(2, activation='softmax'))
    # 編譯: 選擇損失函數、優化方法及成效衡量方式
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    return model

model = build_model()
model.fit(X_train, y_train_hot, batch_size=10, epochs=50, verbose=1, validation_data=(X_test, y_test_hot), callbacks=[tensorboard])    # v2 MOD

# 模型存檔
model_json = model.to_json()
with open("SR_model.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save('SR.h5')  # creates a HDF5 file 'model.h5'


# 預測(prediction)
# mfcc = wav2mfcc('./data/bed/db7c94b3_nohash_1.wav') # j1vde-olshu.wav
# mfcc = wav2mfcc(r'./data/bed/demo3.wav')
# mfcc_reshaped = mfcc.reshape(1, 20, 11, 1)
# print("labels=", get_labels())
# print("predict=", np.argmax(model.predict(mfcc_reshaped)))
