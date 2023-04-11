import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy


PROJECT_DIR = '/Users/nattapolchanpaisit/Documents/GitHub/Algorithm/SC1015-PROJECT/'
class set_b_dataset(Dataset):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.metadata = pd.read_csv(f'{PROJECT_DIR}dataset/set_b.csv')
        self.filenames = []
        for name in self.metadata['fname']: 
            if 'Bunlabelled' not in name: 
                self.filenames.append(name)
        self.dataset = []
        self.fft = []
        for fn in self.filenames:
            data, samplerate = librosa.load(f'{PROJECT_DIR}dataset/{fn}')
            self.dataset.append([data, samplerate])
            self.fft.append(librosa.amplitude_to_db(librosa.stft(data)))
        
        bpm = []
        peak_interval = []
        peak_interval_timer = []
        for i in range(len(self.dataset)):
            peak_array = self.aproximate_bpm(i)
            interval = [(peak_array[i+1] - peak_array[i])/self.dataset[i][1] for i in range(len(peak_array)-1)]
            timer = [peak_array[i]/self.dataset[i][1] for i in range(len(peak_array)-1)]
            val = 60 * len(peak_array) * self.dataset[i][1] / len(self.dataset[i][0]) / 2
            bpm.append(val)
            peak_interval.append(interval)
            peak_interval_timer.append(timer)
        
        self.metadata['bpm'] = bpm
        self.metadata['peak-interval'] = peak_interval
        self.metadata['peak-interval-timer'] = peak_interval_timer
        
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        return self.dataset[idx][0], self.dataset[idx][1]
        
    def show_wave(self, idx: int, **kwargs) -> librosa.display.AdaptiveWaveplot:
        return librosa.display.waveshow(self.dataset[idx][0], sr=self.dataset[idx][1], **kwargs)
    
    def show_spec(self, idx: int, **kwargs) -> matplotlib.collections.QuadMesh:
        return librosa.display.specshow(self.fft[idx], **kwargs)
    
    def denoise(self) -> None:
        # Add denoise function
        # denoise self.dataset
        # dtype(self.dataset): [magnitude array: float, samplerate: int]
        pass
    
    def aproximate_bpm(self, index):
        peak_array, _ = scipy.signal.find_peaks(self.dataset[index][0], height=0.1, 
                                                distance=0.2*self.dataset[index][1])
        return peak_array