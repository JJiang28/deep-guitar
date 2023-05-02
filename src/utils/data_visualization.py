'''
This file contains a bunch of utility functions for data visualization
Author: Kaniel Vicencio
'''

import matplotlib.pyplot as plt
import numpy as np

import librosa 
import librosa.display
def plot_waveform(y, sr, name):

    librosa.display.waveshow(y, sr=sr)
    plt.title(name)
    plt.xlabel('Time / second')
    plt.ylabel('Amplitude')

    plt.show()

def plot_frquency(y, sr, name):
    
    k = np.arange(len(y))
    T = len(y)/sr
    freq = k/T

    DATA_0 = np.fft.fft(y)
    abs_DATA_0 = abs(DATA_0)

    plt.title(name)
    plt.plot(freq, abs_DATA_0)
    plt.xlabel('Freq / Hz')
    plt.ylabel('Amplitude / dB')
    plt.xlim([0, 1000])
    plt.show()

def plot_mfcc(y, sr, name):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', y_axis='log')
    plt.title(name)
    plt.show()

def plot_chromagram(y, sr, name):
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chromagram, sr=sr, x_axis='time', y_axis='log')
    plt.title(name)
    plt.show()

def plot_mel_spectrogram(y, sr, name):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=12)
    librosa.display.specshow(mel_spectrogram, sr=sr, x_axis='time', y_axis='log')
    plt.title(name)
    plt.show()