import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display


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
        
    def __len__(self) -> int:
        return len(self.landmarks_frame)

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