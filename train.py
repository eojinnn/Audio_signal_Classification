import librosa
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from scipy.stats import uniform
import pickle
import joblib
import time
start = time.time()
SAMPLING_RATE = 44100
SECOND = 2

def read_file(path):
    list_dir = os.listdir(path)
    file_list = np.zeros(SAMPLING_RATE*SECOND)
    for file in list_dir:
        audio ,sr = librosa.load(path+'/'+file,sr = SAMPLING_RATE)
        file_list = np.vstack((file_list, audio))
        np.delete(file_list, 0,axis = 0)
    return file_list,sr

def mel_spectrogram(audio_file, test_array, test_label,n_fft, n_mels, hop_length, sr, label):
    mel_spect = librosa.feature.melspectrogram(y=audio_file, sr = sr, n_fft = n_fft, hop_length = hop_length,n_mels = n_mels)
    mel_spect = librosa.power_to_db(mel_spect, ref = np.max)
    mel_spect = np.array(mel_spect)

    rows, columns = mel_spect.shape
    
    mel_spect = mel_spect.reshape((1,rows*columns))

    if test_array.size == 0:
        test_array = mel_spect
    else:
        test_array = np.concatenate((test_array,mel_spect),axis = 0)
    
    test_label = np.append(test_label, label)
    return test_array, test_label

def mfcc(audio_file, test_array, test_label,n_fft, n_mels, hop_length, sr, label,):
    mfcc_spec = librosa.feature.mfcc(y=audio_file, sr = sr,n_mels = n_mels, n_fft = n_fft, hop_length = hop_length)
    mfcc_spec = librosa.power_to_db(mfcc_spec, ref = np.max)
    mfcc_spec = np.array(mfcc_spec)

    rows, columns = mfcc_spec.shape
    
    mfcc_spec = mfcc_spec.reshape((1,rows*columns))

    if test_array.size == 0:
        test_array = mfcc_spec
    else:
        test_array = np.concatenate((test_array,mfcc_spec),axis = 0)
    
    test_label = np.append(test_label, label)
    return test_array, test_label

def making_arr_and_lab(func, audio_file,test_array, test_label,label):
    arrays, labels = func(audio_file = audio_file,test_array = test_array,test_label = test_label, n_fft = 4096, n_mels = 64, hop_length = int(sr*0.0315), sr = sr, label = label)  
    return arrays, labels


noise_audio ,sr = read_file('G:\Dropbox\Dropbox\ㄱㅇㅈ\#ai_hub_data/n')
human_audio_10db ,sr = read_file('G:\Dropbox\Dropbox\ㄱㅇㅈ\#ai_hub_data_add\-10dB')

test_array = np.array([])
test_label = np.array([])

for audio in noise_audio:
    test_array, test_label = making_arr_and_lab(func = mfcc,audio_file = audio,test_array = test_array,test_label = test_label,label = 0)
for audio in human_audio_10db:
    test_array, test_label = making_arr_and_lab(func = mfcc,audio_file = audio,test_array = test_array,test_label = test_label,label = 1)


X_train, X_test, y_train, y_test= train_test_split(test_array, test_label, test_size = 0.2, random_state = 0)

np.save('G:\Dropbox\Dropbox\ㄱㅇㅈ\svc_project/train_data/train_data_mfcc_-10dB.npy', X_train)
np.save('G:\Dropbox\Dropbox\ㄱㅇㅈ\svc_project/train_data/train_label_mfcc_-10dB.npy', y_train)
np.save('G:\Dropbox\Dropbox\ㄱㅇㅈ\svc_project/test_data/test_data_mfcc_-10dB.npy',X_test)
np.save('G:\Dropbox\Dropbox\ㄱㅇㅈ\svc_project/test_data/test_label_mfcc_-10dB.npy',y_test)
#데이터셋을 랜덤하게 80%의 훈련셋과 20%의 데이터셋으로 분류
 
# sc = StandardScaler()
# sc.fit(X_train)
 
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)
#데이터 전처리

print(X_train.shape)
print(test_label)

C_range = np.logspace(-2,10,10)
gamma_range = np.logspace(-9,3,10)
param_grid = dict(gamma = gamma_range, C = C_range)
cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
grid = GridSearchCV(SVC(), param_grid= param_grid, cv = cv)
grid.fit(X_train, y_train)
print("The best parameters are %s with a score of %0.2f"
      %(grid.best_params_, grid.best_score_))

svm_model = SVC(C = 10)
print(svm_model.gamma)
svm_model.fit(X_train, y_train) #SVM 훈련

joblib.dump(svm_model, '20230805 svc mel_-10dB.pkl')
# joblib.dump(grid,'20230805 grid mel_-10dB.pkl')

y_pred = svm_model.predict(X_test) #SVM 테스트

print("예측된 라벨:", y_pred)
print("실제의 라벨:", y_test) #0.8
print("prediction accuracy: {:.2f}".format(np.mean(y_pred == y_test))) #예측 정확도
    

# printing gamma
n_mels, hop_length = test_array.shape

# n_features 값: 스펙트로그램의 열 개수
n_features = hop_length

# std 값: 스펙트로그램 값들의 표준 편차
std = np.std(test_array)

# gamma_scale = 1 / (n_features * std ** 2)
# print(gamma_scale)
# finish = time.time()
# print('proceeding time: ', finish -start)

'''
parameters
test_size = 0.2
random_state = 0
n_mels = 64
hop_length = sr의 int 0.0315
n_fft = 4096
(64,64)
'''