import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

path1 = 'C:\spectrogram\위급상황 음성_음향\Training\[원천]2.강도범죄_1 (15)'
path2 = 'C:\spectrogram/130.도시 소리 데이터/01.데이터/1.Training\원천데이터\TS_1.교통소음'
file_list1 = os.listdir(path1)
file_list2 = os.listdir(path2)
file_list = file_list1+file_list2
test_array = np.array([])
test_label = np.array([])


for file_name in file_list:
    if file_name in file_list2:
        audio_file = path2+'/'+file_name
        y, sr = librosa.load(audio_file,sr = 44100)
    else:
        audio_file = path1+'/'+file_name
        y, sr = librosa.load(audio_file,sr = 44100)

    f = sf.SoundFile(audio_file)
    f_sec = f.frames // f.samplerate 
    #전체 오디오 파일의 길이
    
    start_time = 0
    while start_time < f_sec:
        sec = 1
        ny = y[sr*start_time : sr*(sec + start_time)]
        #audio cut

        hop_length = 512
        n_mels = 64

        mel_spect = librosa.feature.melspectrogram(y=ny,sr=sr,n_mels = n_mels, n_fft = 1024,hop_length = hop_length)
        mel_spect = librosa.power_to_db(mel_spect, ref = np.max)
        mel_spect = np.array(mel_spect) #(n_mels, hop_length)
        
        rows, columns = mel_spect.shape
        mel_spect = mel_spect.reshape((1,rows*columns))

        if test_array.size == 0:
            test_array = mel_spect
        else: 
            test_array = np.concatenate((test_array, mel_spect), axis=0)
        #test배열 생성
        
        if audio_file == path1+'/'+file_name:
            test_label = np.append(test_label, 1)
        else:
            test_label = np.append(test_label, 0)
        #test_label 생성

        start_time = start_time + 1

#test 파일 생성

test_array = np.array(test_array)
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

print("예측된 라벨:",y_pred)
print("실제의 라벨:", y_test) #0.8
print("prediction accuracy: {:.2f}".format(np.mean(y_pred == y_test))) #예측 정확도

n_mels, hop_length = mel_spect.shape

# n_features 값: 스펙트로그램의 열 개수
n_features = hop_length

# std 값: 스펙트로그램 값들의 표준 편차
std = np.std(mel_spect)

gamma_scale = 1 / (n_features * std ** 2)
print(gamma_scale)