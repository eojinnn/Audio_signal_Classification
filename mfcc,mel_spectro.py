import librosa
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    mel_spect = librosa.power_to_db(mel_spect, ref = np.max) #주파수의 최대값을 0db로 주파수 데시벨 변환
    mel_spect = np.array(mel_spect)

    rows, columns = mel_spect.shape
    print(mel_spect.shape)
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
    print(mfcc_spec.shape)
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
noise_audio ,sr = read_file('C:\spectrogram\#ai_hub_data_piece/n')
human_audio_10db ,sr = read_file('C:\spectrogram\#ai_hub_data_piece\y')
test_array = np.array([])
test_label = np.array([])
for audio in noise_audio:
    test_array, test_label = making_arr_and_lab(func = mfcc,audio_file = audio,test_array = test_array,test_label = test_label,label = 0)
for audio in human_audio_10db:
    test_array, test_label = making_arr_and_lab(func = mfcc,audio_file = audio,test_array = test_array,test_label = test_label,label = 1)


X_train, X_test, y_train, y_test= train_test_split(test_array, test_label, test_size = 0.2, random_state = 0)  
#데이터셋을 랜덤하게 80%의 훈련셋과 20%의 데이터셋으로 분류
 
sc = StandardScaler()
sc.fit(X_train)
 
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#데이터 전처리

print(X_train.shape)
print(test_label)

svm_model = SVC(C=105, gamma = 0.000000338)
print(svm_model.gamma)
svm_model.fit(X_train, y_train) #SVM 훈련

y_pred = svm_model.predict(X_test) #SVM 테스트

print("예측된 라벨:", y_pred)
print("실제의 라벨:", y_test) #0.8
print("prediction accuracy: {:.2f}".format(np.mean(y_pred == y_test))) #예측 정확도
    
    
#test_size = 0.2
#random_state = 0
#n_mels = 64
#hop_length = sr의 int 0.0315
#n_fft = 4096
#(64,64)