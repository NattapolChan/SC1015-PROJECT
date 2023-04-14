import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy


class set_b_dataset():
    """ dataclass for storing data & preparing stft
    """
    def __init__(self, path: str) -> None:
        self.SR = 22050 # default sampling rate
        self.metadata = pd.read_csv(f'{path}dataset/set_b.csv')
        self.filenames = []
        for name in self.metadata['fname']: 
            if 'Bunlabelled' not in name:
                self.filenames.append(name)
        self.dataset = []
        self.stft = []
        duration = []
        bpm = []
        peak_interval = []
        peak_interval_timer = []
        
        """preprocess - WAVE & STFT
            -> wave array: np.ndarray, (nframes)
            -> stft: np.ndarray, (freq_range * 10, nframes // 511)
        """
        for i, fn in enumerate(self.filenames):
            data, _ = librosa.load(f'{path}dataset/{fn}')
            self.dataset.append(data)
            self.stft.append(librosa.amplitude_to_db(np.abs(librosa.stft(data))))
            duration.append(len(data)/self.SR)
            
        """preprocess - HRV & BPM
            -> peak array: np.ndarray, (nframes - 1)
            -> peak interval | HRV: np.ndarray, ()
        """
        for i, fn in enumerate(self.filenames):
            peak_array = self.aproximate_bpm(i)
            bpm.append(60 * len(peak_array) * self.SR / len(self.dataset[i]) / 2)
            peak_interval.append([(peak_array[i+1] - peak_array[i])/self.SR for i in range(len(peak_array)-1)])
            peak_interval_timer.append([peak_array[i]/self.SR for i in range(len(peak_array)-1)])
        
        self.metadata['bpm'] = bpm
        self.metadata['peak-interval'] = peak_interval
        self.metadata['peak-interval-timer'] = peak_interval_timer
        self.metadata['duration'] = duration
        
    def __len__(self) -> int:
        return len(self.dataset)
        
    def show_wave(self, idx: int, **kwargs) -> librosa.display.AdaptiveWaveplot:
        return librosa.display.waveshow(self.dataset[idx], sr=self.SR, **kwargs)
    
    def show_spec(self, idx: int, **kwargs) -> matplotlib.collections.QuadMesh:
        return librosa.display.specshow(self.stft[idx], sr=self.SR, **kwargs)
    
    def denoise(self) -> None:
        # Add denoise function
        # denoise self.dataset
        # dtype(self.dataset): [magnitude array: float, samplerate: int]
        pass
    
    def aproximate_bpm(self, index) -> np.ndarray:
        peak_array, _ = scipy.signal.find_peaks(self.dataset[index], height=0.1, 
                                                distance=0.2*self.SR)
        return peak_array
    
class set_b_dataclass(Dataset):
    """ Custom dataset class for feeding stft into torch.utils.data.DataLoader
    """
    def __init__(self, path: str, output_width=20, stft_low = 200, stft_high = 400, list_id = tuple([i for i in range(461)])) -> None:
        super().__init__()
        self.SR = 22050 # default sampling rate
        self.W = output_width
        self.metadata = pd.read_csv(f'{path}dataset/set_b.csv')
        self.labels = []
        self.filenames = []
        self.stft_low = stft_low
        self.stft_high = stft_high
        self.dataset = []
        self.stft = []
        
        for name in self.metadata['fname']: 
            if 'Bunlabelled' not in name:
                self.filenames.append(name)        
        
        """preprocess - WAVE & STFT
            -> wave array: np.ndarray, (nframes)
            -> stft: np.ndarray, (freq_range * 10, nframes // 511)
        """
        for i, fn in enumerate(self.filenames):
            if i in list_id:
                data, _ = librosa.load(f'{path}dataset/{fn}')
                self.dataset.append(data)
                self.stft.append(librosa.amplitude_to_db(np.abs(librosa.stft(data))))
                self.labels.append(self.metadata['label'][i])
        
    def __len__(self) -> int:
        return len(self.stft)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        fft_torch = torch.Tensor(self.stft[idx])
        fft_torch = fft_torch[self.stft_low//10:self.stft_high//10,:]
        fft_torch = F.pad(fft_torch, (0, self.W-fft_torch.size(1)), "constant", 0)
        return fft_torch, self.labels[idx]
        
    def show_wave(self, idx: int, **kwargs) -> librosa.display.AdaptiveWaveplot:
        return librosa.display.waveshow(self.dataset[idx], sr=self.SR, **kwargs)
    
    def show_spec(self, idx: int, **kwargs) -> matplotlib.collections.QuadMesh:
        return librosa.display.specshow(self.stft[idx], sr=self.SR, **kwargs)
    
    def denoise(self) -> None:
        # Add denoise function
        # denoise self.dataset
        # dtype(self.dataset): [magnitude array: float, samplerate: int]
        pass