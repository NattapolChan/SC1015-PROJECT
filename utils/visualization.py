"""
    all functions below plot the figure and return the plot
"""

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import scipy

# load audio from filepath
def get_audio(filepath):
    sound, sample_rate = librosa.load(filepath)
    return ipd.Audio(sound, rate = sample_rate), sound, sample_rate

# display raw signals
# X-axis: time
# Y-axis: amplitude
def show_waveform(sound, sample_rate):
    plt.figure(figsize = (20, 5))
    librosa.display.waveshow(sound, sr=sample_rate)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Signal Waveform')
    plt.show()

# display sound spectrum
# DFT using FFT
def show_spectrum(sound, sample_rate):
    fft = np.fft.fft(sound)
    amp = np.abs(fft) # amplitude of fourier
    t = np.linspace(0, sample_rate, len(amp))

    plt.figure(figsize = (9, 6))
    plt.plot(t[:len(t)//2], amp[:len(t)//2]) # remove complex conjugate from fft
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Spectrum')
    plt.show()

# display linear frequency power spectogram
# STFT
def show_spectrogram(sound, sample_rate, hop_length):
    stft = librosa.stft(sound, hop_length=hop_length)
    spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    plt.figure(figsize=(12, 8))
    librosa.display.specshow(spec, sr = sample_rate, x_axis = 'time', y_axis = 'linear', hop_length=hop_length)
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.colorbar(format = '%2.0f dB')
    plt.title('Spectrogram')
    plt.show()

# mel-scale spectrogram
def show_mel_spectrogram(sound, sample_rate, hop_length):
    mel_spec = librosa.feature.melspectrogram(y = sound, sr = sample_rate, hop_length=hop_length)
    
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(mel_spec, sr = sample_rate, x_axis = 'time', y_axis='mel', hop_length=hop_length)
    plt.colorbar(format = '%2.0f')
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.title('Mel Spectrogram')
    plt.show()

    return mel_spec    

# log mel spectrogram
def show_log_mel_spectrogram(mel_spec, sample_rate, hop_length):
    log_mel_spec = librosa.power_to_db(mel_spec, ref = np.max)

    plt.figure(figsize=(12, 8))
    librosa.display.specshow(log_mel_spec, sr = sample_rate, x_axis = 'time', y_axis='mel', hop_length=hop_length)
    plt.colorbar(format = '%2.0f dB')
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.title('Log Mel Spectrogram')
    plt.show()

# zero-phase butterworth filter of type band-pass
# low, high: cutoffs in Hz
def band_pass_filter(sound, low, high, order = 3, sample_rate = 22050):
    low /= sample_rate/2
    high /= sample_rate/2

    b, a = scipy.signal.butter(order, (low, high), 'band')
    filtered = scipy.signal.filtfilt(b, a, sound)

    return filtered