"""
1. rate 구하기
2. snr값에 따라 rate달라지게 하기
rate곱해서 두신호 합성
"""

import librosa
import os
import math
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

def audio_file_dir(path, file_index):
    audio_file_list = os.listdir(path)
    audio_file = path + '/' + audio_file_list[file_index]
    return audio_file
#파일 불러오기

def cal_snr_rate(noise, human, snr):
    # Check if there are any zero values in the 'noise' array
    if np.any(noise == 0):
        # Replace zero values with a small positive value to avoid division by zero
        noise = np.where(noise == 0, 1e-9, noise)

    rate = np.abs(np.sqrt(np.square(human) * snr / np.square(noise)))
    return rate

def normalize_audio(audio_data):
    mean_value = np.mean(audio_data)
    std_value = np.std(audio_data)
    normalized_audio = (audio_data-mean_value) / std_value
    return normalized_audio

def mel_spectrogram(noisy_audio,test_array,test_label,label, hop_length, n_mels):
    mel_spect = librosa.feature.melspectrogram(y=noisy_audio,sr=44100,n_mels = n_mels, n_fft = 1024,hop_length = hop_length)
    mel_spect = librosa.power_to_db(mel_spect, ref = np.max)
    mel_spect = np.array(mel_spect) #(n_mels, hop_length)
    
    rows, columns = mel_spect.shape
    mel_spect = mel_spect.reshape((1,rows*columns))

    if test_array.size == 0:
        test_array = mel_spect
    else: 
        test_array = np.concatenate((test_array, mel_spect), axis=0)
    #test배열 생성
    
    test_label = np.append(test_label, label)
    return test_array, test_label

path_human = 'C:/spectrogram\cmu_us_clb_arctic-0.95-release'
path_noise = 'C:/spectrogram/130.도시 소리 데이터/01.데이터/1.Training\원천데이터\TS_1.교통소음'

test_array = np.array([])
test_label = np.array([])

for x in range(1000):
    try:
        noise_audio, sr = librosa.load(audio_file_dir(path_noise, x), sr=44100)
        human_audio, sr = librosa.load(audio_file_dir(path_human, x), sr=44100)
        
        num_segments = min(len(human_audio), len(noise_audio)) // 44100
        # 오디오 파일의 총 길이
        print(num_segments)
        
        for start_time in range(num_segments):
            sec = 1
            sec_h = human_audio[44100 * start_time: 44100 * (start_time + sec)]
            sec_n = noise_audio[44100 * start_time: 44100 * (start_time + sec)]

            sec_n_adj = cal_snr_rate(sec_n,sec_h,10)

            noisy_audio = sec_h + sec_n_adj
            # 음성 
        
            normal_audio = normalize_audio(noisy_audio)

            test_array, test_label = mel_spectrogram(normal_audio,test_array, test_label,label = 1, hop_length=512,n_mels=64)

            start_time = start_time + 1
    except Exception as e:
        break
print(test_label)

# num_segments = len(noise_audio) // 44100
# # 오디오 파일의 총 길이
# for start_time in range(num_segments):
#     sec = 1
#     sec_n = noise_audio[44100 * start_time: 44100 * (start_time + sec)]

#     noisy_audio = sec_n
#     # 음성 합성
#     normal_audio = normalize_audio(noisy_audio)

#     test_array, test_label = mel_spectrogram(normal_audio,test_array, test_label,label = 0, hop_length=512,n_mels=64)

#     start_time = start_time + 1

#test 파일 생성
for x in range(1000):
    try:
        noise_audio, sr = librosa.load(audio_file_dir(path_noise, x), sr=44100)
        human_audio, sr = librosa.load(audio_file_dir(path_human, x), sr=44100)
        
        num_segments = min(len(human_audio), len(noise_audio)) // 44100
        # 오디오 파일의 총 길이
        
        for start_time in range(num_segments):
            sec = 1
            sec_n = noise_audio[44100 * start_time: 44100 * (start_time + sec)]

            noisy_audio = sec_n
            # 음성 
            normal_audio = normalize_audio(noisy_audio)

            test_array, test_label = mel_spectrogram(normal_audio,test_array, test_label,label = 0, hop_length=512,n_mels=64)

            start_time = start_time + 1
    except Exception as e:
        break

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

