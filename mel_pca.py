from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
import os
import time
import librosa
import numpy as np
import wave,math
from scipy.io import wavfile 
import soundfile as sf
import matplotlib.pylab as plt
import padasip as pa
from sklearn.decomposition import PCA

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
# SIR x_db ndarray
dy = np.load('C:\spectrogram/datas/0db_dy.npy') # lms 결과, I
ddy = np.load('C:\spectrogram/datas/0db_ddy.npy') # 원본신호 X+I
dn = np.load('C:\spectrogram/datas/dn.npy') # lms 결과, I
ddn = np.load('C:\spectrogram/datas/ddn.npy') # 원본신호 X+I

test_array = np.array([])
test_label = np.array([])


for i in range(len(dy)):
    # label 1
    # 데이터 전처리
    y = dy[i].reshape((-1,1))
    # print(y.shape)
    data_y = ddy[i].reshape((ddy[i].shape[0], 1))
    # print(min(data_y), max(data_y))
    data_y = data_y / max(abs(data_y)) # 정규화

    # X에 가우시안 씌우기
    x_awgn, n_awgn = awgn(data_y-y,10)
    
    sr = 44100
    test_label = np.append(test_label, 1)
    y_adj = y.flatten()
    mel_spect = librosa.feature.melspectrogram(y=y_adj, sr = sr, n_fft = 4096, hop_length = int(sr*0.0315), n_mels = 64)
    mel_spect = librosa.power_to_db(mel_spect, ref = np.max)
    mel_spect = np.array(mel_spect)

    #mel파일 저장 X signal
    #filename = f"C:\spectrogram/mel_image_processing/0db_10db/1/mel_0_0_{i}.npy"  # 저장할 파일명 생성
    #np.save(filename, mel_spect)  # 저장할 파일명 생성

    mel_spect = pca_t(mel_spect)
    if test_array.size == 0:
        test_array = mel_spect
    else:
        test_array = np.concatenate((test_array,mel_spect),axis = 0)

    #label 0
    #noise입력 noise제거
    y = dn[i].reshape((-1,1))
    data_y = ddn[i].reshape((ddn[i].shape[0], 1))
    data_y = data_y / max(abs(data_y)) # 정규화

    # (N-N)에 가우시안 씌우기
    x_awgn, n_awgn = awgn(data_y-y,10)
    
    sr = 44100
    test_label = np.append(test_label, 0)
    y_adj = y.flatten()
    mel_spect = librosa.feature.melspectrogram(y=y_adj, sr = sr, n_fft = 4096, hop_length = int(sr*0.0315), n_mels = 64)
    mel_spect = librosa.power_to_db(mel_spect, ref = np.max)
    mel_spect = np.array(mel_spect)

    mel_spect = pca_t(mel_spect)
    if test_array.size == 0:
        test_array = mel_spect
    else:
        test_array = np.concatenate((test_array,mel_spect),axis = 0)

    #mel파일 저장 X signal
    #filename = f"C:\spectrogram/mel_image_processing/0db_10db/0/mel_0_0_{i}.npy"  # 저장할 파일명 생성
    #np.save(filename, mel_spect)  # 저장할 파일명 생성

np.save('C:\spectrogram/mel+pca/test_array/pca_0_10t.npy', test_array)
np.save('C:\spectrogram/mel+pca/test_label/pca_0_10t.npy', test_label)
print('done')



