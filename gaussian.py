import librosa
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from scipy.stats import uniform
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import joblib
import time
start = time.time()


noise_audio = np.load('G:\Dropbox\Dropbox\ㄱㅇㅈ/feature_mel/n_0dB_awgn.npy')
human_audio_10db = np.load('G:\Dropbox\Dropbox\ㄱㅇㅈ/feature_mel/y_-10dB_awgn.npy')

test_array = np.concatenate((noise_audio, human_audio_10db),axis = 0)
test_label = np.concatenate((np.zeros((4960,1)),np.ones((4960,1))),axis =None)

test_array = test_array.reshape(len(test_array),64*64)
X_train, X_test, y_train, y_test= train_test_split(test_array, test_label, test_size = 0.2, random_state = 0)

print(X_train.shape)

print(test_label)


svm_model = SVC(C = 100)
print(svm_model.gamma)
svm_model.fit(X_train, y_train) #SVM 훈련


y_pred = svm_model.predict(X_test) #SVM 테스트

print("예측된 라벨:", y_pred)
print("실제의 라벨:", y_test) #0.8
print("prediction accuracy: {:.4f}".format(np.mean(y_pred == y_test))) #예측 정확도
ConfusionMatrixDisplay.from_predictions(y_pred, y_test)
plt.show()
    

# printing gamma
n_mels, hop_length = test_array.shape

# n_features 값: 스펙트로그램의 열 개수
n_features = hop_length

# std 값: 스펙트로그램 값들의 표준 편차
std = np.std(test_array)

gamma_scale = 1 / (n_features * std ** 2)
print(gamma_scale)
finish = time.time()
print('proceeding time: ', finish -start)

'''
parameters
test_size = 0.2
random_state = 0
n_mels = 64
hop_length = sr의 int 0.0315
n_fft = 4096
(64,64)
'''