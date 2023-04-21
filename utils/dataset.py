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
        
    def show_wave(self, idx: int, flatten=False, **kwargs) -> librosa.display.AdaptiveWaveplot:
        sound_flatten = ((self.dataset[idx] > 1.5 * np.std(self.dataset[idx])) + (self.dataset[idx] < -1.5 * np.std(self.dataset[idx]))) * self.dataset[idx]
        if flatten:
            return librosa.display.waveshow(sound_flatten, sr=self.SR, **kwargs)
        return librosa.display.waveshow(self.dataset[idx], sr=self.SR, **kwargs)
    
    def show_spec(self, idx: int, **kwargs) -> matplotlib.collections.QuadMesh:
        return librosa.display.specshow(self.stft[idx], sr=self.SR, **kwargs)
    
    def aproximate_bpm(self, index) -> np.ndarray:
        peak_array, _ = scipy.signal.find_peaks(self.dataset[index], height=0.1, 
                                                distance=0.2*self.SR)
        return peak_array
    
class set_b_dataclass(Dataset):
    """ Custom dataset class for feeding stft into torch.utils.data.DataLoader
    """
    def __init__(self, path: str, output_width=200, stft_low = 200, 
                 stft_high = 400, list_id = tuple([i for i in range(461)]),
                 oversample = False, out_classes = list(('normal', 'murmur', 'extrastole')),
                 denoise=True) -> None:
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
        
        self.count_normal = []
        self.count_murmur = []
        self.count_extrastole = []
        for i, fn in enumerate(self.filenames):
            if i in list_id:
                data, _ = librosa.load(f'{path}dataset/{fn}')
                self.dataset.append(data)
                if denoise:
                    self.dataset[-1] = self.denoiser(data)
                stft = librosa.amplitude_to_db(np.abs(librosa.stft(self.dataset[-1], n_fft=2048)))
                stft_torch = torch.Tensor(stft)
                # shift window resampling
                while (stft_torch.size(1) >= 200):
                    self.stft.append(stft_torch[:, :200])
                    stft_torch = stft_torch[:, 70:]
                    self.labels.append(self.metadata['label'][i])
                
                if self.metadata['label'][i] in out_classes:
                    self.stft.append(stft_torch)
                    self.labels.append(self.metadata['label'][i])
                
                if self.labels[-1] == 'normal': 
                    self.count_normal.append(len(self.labels)-1)
                elif self.labels[-1] == 'murmur' and 'murmur' in out_classes: 
                    self.count_murmur.append(len(self.labels)-1)
                    if oversample: 
                        self.augment_minority_classes()
                        self.augment_minority_classes()
                elif self.labels[-1] == 'extrastole' and 'extrastole' in out_classes: 
                    self.count_extrastole.append(len(self.labels)-1)
                    if oversample: 
                        self.augment_minority_classes()
                        self.augment_minority_classes()
                        self.augment_minority_classes()
                
        self.count_class = pd.DataFrame()
        self.count_class['count'] = [len(self.count_normal), len(self.count_murmur), len(self.count_extrastole)]
        self.count_class['index'] = [self.count_normal, self.count_murmur, self.count_extrastole]
        self.count_class['label'] = ['normal', 'murmur', 'extrastole']
        
        print('Number of data per class')
        print(self.count_class['count'])
        
    def __len__(self) -> int:
        return len(self.stft)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        fft_torch = self.stft[idx] + torch.Tensor(np.random.normal(loc=0.00, scale=1.0, size=(self.stft[idx].shape[0], self.stft[idx].shape[1])))
        fft_torch = fft_torch[self.stft_low//5:self.stft_high//5,:]
        fft_torch = F.pad(fft_torch, (0, self.W-fft_torch.size(1)), "constant", 0)
        label = 0 if self.labels[idx]=='normal' else 1 if self.labels[idx]=='murmur' else 2
        fft_torch = (fft_torch - 15) / 25
        return fft_torch, label
        
    def show_wave(self, idx: int, **kwargs) -> librosa.display.AdaptiveWaveplot:
        sound_flatten = ((self.dataset[idx] > 0.15) + (self.dataset[idx] < -0.15)) * self.dataset[idx]
        return librosa.display.waveshow(sound_flatten, sr=self.SR, **kwargs)
    
    def show_spec(self, idx: int, **kwargs) -> matplotlib.collections.QuadMesh:
        return librosa.display.specshow(self.stft[idx], sr=self.SR, **kwargs)
    
    """event rate: murmur = 100/461, extrastole = 50/461
        oversampling: murmur x 2, extrastole x 4
    """
    def augment_minority_classes(self) -> None:
        stft = self.stft[-1]
        stft += np.random.normal(loc=0.0, scale=3.0, size=(stft.shape[0], stft.shape[1]))
        self.stft.append(stft)
        if self.labels[-1] == 'murmur':
            self.labels.append('murmur')
            self.count_murmur.append(len(self.labels)-1)
        else:
            self.labels.append('extrastole')
            self.count_extrastole.append(len(self.labels)-1)
    
    def fft_denoiser(self, x, n_components, to_real=True):
        n = len(x)
        fft = np.fft.fft(x, n)
        PSD = fft * np.conj(fft) / n
        _mask = PSD > n_components
        fft = _mask * fft
        clean_data = np.fft.ifft(fft)
        if to_real:
            clean_data = clean_data.real
        return clean_data

    def denoiser(self, wave:np.ndarray) -> np.ndarray:
        noised_data = self.fft_denoiser(wave, 0.001)
        sos = scipy.signal.butter(1, 250, 'hp', fs=22050, output='sos')
        filtered_noised_data = scipy.signal.sosfilt(sos, noised_data)
        return filtered_noised_data