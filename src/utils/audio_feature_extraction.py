'''
This file contains functions for audio feature extractions
Author: Kaniel Vicencio
'''

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Waveform
def get_waveform(fpath): 
    signal, sample_rate = librosa.load(fpath)
    return signal, sample_rate

# feature extraction: mel-freq cepstral coefficients 
def get_mfccs(y, sr):
    df_mfccs = pd.DataFrame()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    norm = mfccs / np.linalg.norm(mfccs)

    for n_mfcc in range(len(norm)):
        df_mfccs['MFCC_%d'%(n_mfcc+1)] = norm.T[n_mfcc]

    return df_mfccs, norm

# feature extraction: mel spectogram

def get_melspectrogram(y, sr):
    df_melspectrogram = pd.DataFrame()
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=12)
    norm = mel_spectrogram / np.linalg.norm(mel_spectrogram)

    for n_mel in range(len(norm)):
        df_melspectrogram['Mel_Spectogram_%d'%(n_mel+1)] = norm.T[n_mel]

    return df_melspectrogram, norm

# feature extraction: chroma
def get_chromagram(y, sr):
    df_chroma = pd.DataFrame()
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    norm = chromagram / np.linalg.norm(chromagram)

    for n_chroma in range(len(norm)):
        df_chroma['Chroma_%d'%(n_chroma+1)] = norm.T[n_chroma]

    return df_chroma, norm

# generate a feature matix with the 3 features 
def generate_feature_matrix(df_mfccs, df_chroma, df_melspectrogram):
    feature_matrix = pd.concat([df_mfccs, df_chroma, df_melspectrogram], axis=1)
    return feature_matrix

