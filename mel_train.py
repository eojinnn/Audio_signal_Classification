from scipy.io import wavfile
import numpy as np
import os
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
import librosa
from sklearn.decomposition import PCA

path_x = "C:/spectrogram/#ai_hub_data/x/"
dir_list_x = os.listdir(path_x)
wavfile_x = [file for file in dir_list_x if file.endswith(".wav")]
test_label = np.array([])
test_array = np.array([])

def pca_t(mel_spectrogram):
    pca = PCA(n_components=1) # 주성분을 몇개로 할지 결정
    principalComponents = pca.fit_transform(mel_spectrogram)
    principalComponents = np.transpose(principalComponents)
    return principalComponents

def awgn(s,SNRdB,L=1):
        """
        AWGN channel
        Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
        returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
        Parameters:
            s : input/transmitted signal vector
            SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
            L : oversampling factor (applicable for waveform simulation) default L = 1.
        Returns:
            r : received signal vector (r=s+n)
        """
        gamma = 10**(SNRdB/10) #SNR to linear scale
        if s.ndim==1:# if s is single dimensional vector
            P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
        else: # multi-dimensional signals like MFSK
            P=L*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
        N0=P/gamma # Find the noise spectral density
        if isrealobj(s):# check if input is real/complex object type
            n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
        else:
            n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
        r = s + n # received signal
        return r,n   


for i in range(len(wavfile_x)):
    fs,data_x = wavfile.read(path_x + wavfile_x[i]) # input matrix
    # print(path_x + wavfile_x[i])
    # data_x= data_x.astype(np.float32)
    # print(i)
    # print(data_x.shape[0])
    data_x = data_x.reshape((data_x.shape[0], 1))
    # print(min(data_x), max(data_x))
    data_x = data_x / max(abs(data_x))
    # print(min(data_x), max(data_x))
    # print(data_x.shape)

    x_awgn, n_awgn = awgn(data_x,0)
    tem = [x_awgn, n_awgn]
    # sf.write("C:/spectrogram/awgn_audio.wav", x_awgn, 44100, format='WAV')
    for y in tem:
        sr = 44100
        if (y == x_awgn).all():    
            test_label = np.append(test_label, 1)
        else:
            test_label = np.append(test_label, 0)
        y_adj = y.flatten()
        mel_spect = librosa.feature.melspectrogram(y=y_adj, sr = sr, n_fft = 4096, hop_length = int(sr*0.0315), n_mels = 64)
        mel_spect = librosa.power_to_db(mel_spect, ref = np.max)
        mel_spect = np.array(mel_spect)

        rows, columns = mel_spect.shape
        # print(mel_spect.shape)
        mel_spect = pca_t(mel_spect)

        if test_array.size == 0:
            test_array = mel_spect
        else:
            test_array = np.concatenate((test_array,mel_spect),axis = 0)

for i in range(len(wavfile_x)):
    fs,data_x = wavfile.read(path_x + wavfile_x[i]) # input matrix
    # print(path_x + wavfile_x[i])
    # data_x= data_x.astype(np.float32)
    # print(i)
    # print(data_x.shape[0])
    data_x = data_x.reshape((data_x.shape[0], 1))
    # print(min(data_x), max(data_x))
    data_x = data_x / max(abs(data_x))
    # print(min(data_x), max(data_x))
    # print(data_x.shape)

    x_awgn, n_awgn = awgn(data_x,-10)
    tem = [x_awgn, n_awgn]
    # sf.write("C:/spectrogram/awgn_audio.wav", x_awgn, 44100, format='WAV')
    for y in tem:
        sr = 44100
        if (y == x_awgn).all():    
            test_label = np.append(test_label, 1)
        else:
            test_label = np.append(test_label, 0)
        y_adj = y.flatten()
        mel_spect = librosa.feature.melspectrogram(y=y_adj, sr = sr, n_fft = 4096, hop_length = int(sr*0.0315), n_mels = 64)
        mel_spect = librosa.power_to_db(mel_spect, ref = np.max)
        mel_spect = np.array(mel_spect)

        rows, columns = mel_spect.shape
        # print(mel_spect.shape)
        mel_spect = pca_t(mel_spect)

        if test_array.size == 0:
            test_array = mel_spect
        else:
            test_array = np.concatenate((test_array,mel_spect),axis = 0)

for i in range(len(wavfile_x)):
    fs,data_x = wavfile.read(path_x + wavfile_x[i]) # input matrix
    # print(path_x + wavfile_x[i])
    # data_x= data_x.astype(np.float32)
    # print(i)
    # print(data_x.shape[0])
    data_x = data_x.reshape((data_x.shape[0], 1))
    # print(min(data_x), max(data_x))
    data_x = data_x / max(abs(data_x))
    # print(min(data_x), max(data_x))
    # print(data_x.shape)

    x_awgn, n_awgn = awgn(data_x,-5)
    tem = [x_awgn, n_awgn]
    # sf.write("C:/spectrogram/awgn_audio.wav", x_awgn, 44100, format='WAV')
    for y in tem:
        sr = 44100
        if (y == x_awgn).all():    
            test_label = np.append(test_label, 1)
        else:
            test_label = np.append(test_label, 0)
        y_adj = y.flatten()
        mel_spect = librosa.feature.melspectrogram(y=y_adj, sr = sr, n_fft = 4096, hop_length = int(sr*0.0315), n_mels = 64)
        mel_spect = librosa.power_to_db(mel_spect, ref = np.max)
        mel_spect = np.array(mel_spect)

        rows, columns = mel_spect.shape
        # print(mel_spect.shape)
        mel_spect = pca_t(mel_spect)

        if test_array.size == 0:
            test_array = mel_spect
        else:
            test_array = np.concatenate((test_array,mel_spect),axis = 0)

for i in range(len(wavfile_x)):
    fs,data_x = wavfile.read(path_x + wavfile_x[i]) # input matrix
    # print(path_x + wavfile_x[i])
    # data_x= data_x.astype(np.float32)
    # print(i)
    # print(data_x.shape[0])
    data_x = data_x.reshape((data_x.shape[0], 1))
    # print(min(data_x), max(data_x))
    data_x = data_x / max(abs(data_x))
    # print(min(data_x), max(data_x))
    # print(data_x.shape)

    x_awgn, n_awgn = awgn(data_x,0)
    tem = [x_awgn, n_awgn]
    # sf.write("C:/spectrogram/awgn_audio.wav", x_awgn, 44100, format='WAV')
    for y in tem:
        sr = 44100
        if (y == x_awgn).all():    
            test_label = np.append(test_label, 1)
        else:
            test_label = np.append(test_label, 0)
        y_adj = y.flatten()
        mel_spect = librosa.feature.melspectrogram(y=y_adj, sr = sr, n_fft = 4096, hop_length = int(sr*0.0315), n_mels = 64)
        mel_spect = librosa.power_to_db(mel_spect, ref = np.max)
        mel_spect = np.array(mel_spect)

        rows, columns = mel_spect.shape
        # print(mel_spect.shape)
        mel_spect = pca_t(mel_spect)

        if test_array.size == 0:
            test_array = mel_spect
        else:
            test_array = np.concatenate((test_array,mel_spect),axis = 0)

for i in range(len(wavfile_x)):
    fs,data_x = wavfile.read(path_x + wavfile_x[i]) # input matrix
    # print(path_x + wavfile_x[i])
    # data_x= data_x.astype(np.float32)
    # print(i)
    # print(data_x.shape[0])
    data_x = data_x.reshape((data_x.shape[0], 1))
    # print(min(data_x), max(data_x))
    data_x = data_x / max(abs(data_x))
    # print(min(data_x), max(data_x))
    # print(data_x.shape)

    x_awgn, n_awgn = awgn(data_x,5)
    tem = [x_awgn, n_awgn]
    # sf.write("C:/spectrogram/awgn_audio.wav", x_awgn, 44100, format='WAV')
    for y in tem:
        sr = 44100
        if (y == x_awgn).all():    
            test_label = np.append(test_label, 1)
        else:
            test_label = np.append(test_label, 0)
        y_adj = y.flatten()
        mel_spect = librosa.feature.melspectrogram(y=y_adj, sr = sr, n_fft = 4096, hop_length = int(sr*0.0315), n_mels = 64)
        mel_spect = librosa.power_to_db(mel_spect, ref = np.max)
        mel_spect = np.array(mel_spect)

        rows, columns = mel_spect.shape
        # print(mel_spect.shape)
        mel_spect = pca_t(mel_spect)

        if test_array.size == 0:
            test_array = mel_spect
        else:
            test_array = np.concatenate((test_array,mel_spect),axis = 0)

for i in range(len(wavfile_x)):
    fs,data_x = wavfile.read(path_x + wavfile_x[i]) # input matrix
    # print(path_x + wavfile_x[i])
    # data_x= data_x.astype(np.float32)
    # print(i)
    # print(data_x.shape[0])
    data_x = data_x.reshape((data_x.shape[0], 1))
    # print(min(data_x), max(data_x))
    data_x = data_x / max(abs(data_x))
    # print(min(data_x), max(data_x))
    # print(data_x.shape)

    x_awgn, n_awgn = awgn(data_x,10)
    tem = [x_awgn, n_awgn]
    # sf.write("C:/spectrogram/awgn_audio.wav", x_awgn, 44100, format='WAV')
    for y in tem:
        sr = 44100
        if (y == x_awgn).all():    
            test_label = np.append(test_label, 1)
        else:
            test_label = np.append(test_label, 0)
        y_adj = y.flatten()
        mel_spect = librosa.feature.melspectrogram(y=y_adj, sr = sr, n_fft = 4096, hop_length = int(sr*0.0315), n_mels = 64)
        mel_spect = librosa.power_to_db(mel_spect, ref = np.max)
        mel_spect = np.array(mel_spect)

        rows, columns = mel_spect.shape
        # print(mel_spect.shape)
        mel_spect = pca_t(mel_spect)

        if test_array.size == 0:
            test_array = mel_spect
        else:
            test_array = np.concatenate((test_array,mel_spect),axis = 0)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import joblib
import time
start = time.time()
print(test_array.shape)

#test_array = test_array.reshape(len(test_array),64*64)
print(test_array.shape)
print(test_label)
print(test_label.shape)


svm_model = SVC(C = 100)
print(svm_model.gamma)
svm_model.fit(test_array, test_label) #SVM 훈련

joblib.dump(svm_model, './mel+pca/pkl/mel+pca.pkl')

n_mels, hop_length = test_array.shape
n_features = hop_length
std = np.std(test_array)

gamma_scale = 1 / (n_features * std ** 2)
print(gamma_scale)
finish = time.time()
print('proceeding time: ', finish - start)
    
    

