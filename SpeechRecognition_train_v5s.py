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

def build_model(hp):  # random search passes this hyperparameter() object 
    
    #建立簡單的線性執行的模型
    model = Sequential()
    # 建立卷積層，filter=32,即 output size, Kernal Size: 2x2, activation function 採用 relu
    model.add(Conv2D(hp.Int('input_units',min_value=32,max_value=256,step=32), kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
    # 建立池化層，池化大小=2x2，取最大值
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int('n_layers',1,4)):
        model.add(Conv2D(hp.Int(f'conv_{i}_units',min_value=32,max_value=256,step=32),(2,2)))
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


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=50,  # how many model variations to test?
    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
    directory=LOG_DIR)

# tuner.search(x=X_train,y=y_train_hot,epochs=10,batch_size=10,validation_data=(X_test, y_test_hot))

# with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
#     pickle.dump(tuner, f)

# Load pickle to see best model result
tuner = pickle.load(open("tuner_1653570868.pkl","rb"))

tuner.get_best_hyperparameters()[0].values
tuner.results_summary()
tuner.get_best_models()[0].summary()
