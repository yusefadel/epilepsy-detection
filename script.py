import zipfile
import mne
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, get_window
import wfdb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from scipy.signal import welch
from scipy import signal
from keras.models import load_model


#############################
## 
# Step 1: Remove artifacts and noise using a bandpass filter

def bandpass_filter(signal1, lowcut, highcut, fs):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')
    filtered_signal = lfilter(b, a, signal1, axis=1)
    return filtered_signal
#############################
# Step 2: Segment the EEG signal into shorter epochs or time windows

def segment_signal(signal, window_size, overlap):
    num_samples = signal.shape[1]  # Get the number of data points in the signal
    stride = int(window_size * (1 - overlap))
    num_windows = int((num_samples - window_size) / stride) + 1

    segmented_data = np.zeros((num_windows, window_size))

    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        segmented_data[i, :] = signal[0, start:end]

    return segmented_data
#############################
# Step 3: Apply a windowing function to each epoch to reduce spectral leakage

def apply_window(segmented_data, window_type='hamming'):
    num_windows, window_size = segmented_data.shape
    window = get_window(window_type, window_size)

    for i in range(num_windows):
        segmented_data[i, :] = segmented_data[i, :] * window

    return segmented_data

#############################
# Step 4: Apply the FFT to each windowed epoch
def apply_fft(segmented_data, fs):
    num_windows, window_size = segmented_data.shape
    freqs, psd = welch(segmented_data[0, :], fs, nperseg=window_size)
    num_freqs = len(freqs)
    fft_result = np.zeros((num_windows, num_freqs))

    for i in range(num_windows):
        freqs, psd = welch(segmented_data[i, :], fs, nperseg=window_size)
        fft_result[i, :] = psd

    return freqs, fft_result

##############################
#step 5 : feature extraction
def extract_features(freqs, fft_result, frequency_bands):
    features = []

    for band in frequency_bands:
        band_indices = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        band_power = np.sum(fft_result[:, band_indices], axis=1)
        features.append(band_power)

    return np.vstack(features).T
##############################
#step 6: normalization
def normalize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    normalized_features = (features - mean) / std
    return normalized_features
##########################
def duplicate_signal(signal, target_length):
    original_length = signal.shape[1]
    
    # Check if the target length is a multiple of the original length
    if target_length % original_length != 0:
        raise ValueError("Target length must be a multiple of the original signal length.")

    # Calculate the number of repetitions needed
    repetitions = target_length // original_length
    
    # Use np.tile to repeat the signal along its second axis
    duplicated_signal = np.tile(signal, (1, repetitions))
    
    return duplicated_signal
##########################
def preprocess_signal(signal,lowcut,highcut):
    signal_num_samples = len(signal[0])
    duration_seconds = 10.0  # Adjust the duration based on your segment
    signal_sampling_frequency = signal_num_samples/duration_seconds

    filtered_signal=bandpass_filter(signal,lowcut,highcut,fs=signal_sampling_frequency)
    signal_segmented=segment_signal(filtered_signal,window_size=256,overlap=0.5)
    signal_windowed=apply_window(signal_segmented)
    signal_freqs,signal_data_fft=apply_fft(signal_windowed,fs=signal_sampling_frequency)
    frequency_bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    signal_features=extract_features(signal_freqs,signal_data_fft,frequency_bands)
    signal_normalized=normalize_features(signal_features)
    dup_feature_signal=duplicate_signal(signal_normalized,1000)
    return dup_feature_signal

###############################

###############################
def detect_signal(model, signal, class_labels):
    # Assuming signal is a 1D array

    # Use the model to predict the class probabilities
    probabilities = model.predict(signal)
    # Get the class with the highest probability
    predicted_class = np.argmax(probabilities)
    print(predicted_class)
    predicted_label = class_labels[predicted_class]

    return predicted_label
    return predicted_label
if __name__ == "__main__":
    np.random.seed(42)
    signal = sleep[1:2]
    model = load_model('/content/drive/MyDrive/dataeeg/eeg/modeleeg1.h5')
    signal_processed = preprocess_signal(signal, lowcut=0.5, highcut=4.0)
    
    # Specify your class labels based on your model
    class_labels = ['normal', 'seizure', 'sleep']
    
    prediction = detect_signal(model, signal_processed, class_labels)
    print(prediction)
    