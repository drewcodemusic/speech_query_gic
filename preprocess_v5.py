import librosa
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

DATA_PATH = "./data_speech/"


# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):            # O
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    print('label_indices:', label_indices)
    return labels, label_indices, to_categorical(label_indices)


# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_pad_len=11):            # O
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    
    # cut audio file
    i=0
    # 訓練資料的長度
    wav_length=5334
    # 聲音檔過長，擷取片段
    if len(wave) > wav_length:
        # 尋找最大聲的點，取前後各半
        i=np.argmax(wave)
        if i > (wav_length):
            wave = wave[i-int(wav_length/2):i+int(wav_length/2)]
        else:
            # 聲音檔過長，取前面
            wave = wave[0:wav_length]
    
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width < 0:
        pad_width = 0
        mfcc = mfcc[:,:11]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc


def save_data_to_array(path=DATA_PATH, max_pad_len=11):            # O
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)

def get_train_test(split_ratio=0.6, random_state=42):       # O
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)

