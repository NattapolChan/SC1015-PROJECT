# SC1015-PROJECT-T25
- Chanpaisit Nattapol
- Maison Sapol
- Saeng-nil Natthakan

Time Series Projects

# About
In this project, we examine the [heart beat sounds dataset](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds), with a particular focus on developing the initial screening stage for cardiac pathologies in a hospital setting. Our attention is directed towards dataset B, which consists of heartbeat audio files obtained from clinical trials in hospitals using digital stethoscopes. Our objective is to classify the heartbeat audio into normal, murmur, and extrasystole conditions. Additionally, we have discovered interesting findings by experimenting with features that can contribute to further advancements in this research area.

# Exploratory Data Analysis (EDA)
We found the number of data points in each class: {normal: 320, murmur: 95, extrasystole: 46}. This indicates an imbalance in the data that needs to be addressed. We visualized the sound waveform, frequency spectrum, and power spectrogram. Additionally, we identified noise present in some audio files, which requires cleaning.
# Preprocess
We employ Fast Fourier Transform (FFT) to denoise audio files by eliminating specific frequencies that persist in a power spectral density. The visualizations of the Short-Time Fourier Transform (STFT) before and after denoising can be seen in the preprocessing file. Additionally, we perform data augmentation for the underrepresented classes (murmur and extrasystole) by introducing low intensity white noise.
# Machine Learning models
We experimented with two architectures: ResNet and InceptionTime. We found that InceptionTime generally provides better performance while containing fewer parameters, resulting in shorter training times. As a result, all of our models are based on the InceptionTime architecture. We attempted to feed the model with audio of the entire length, but the results were unsatisfactory, and it took an enormous amount of time to train the model due to the excessive length of the features. Next, we experimented with features fed into the model using the Librosa library, such as Mel-frequency cepstral coefficients (MFCCs), Mel spectrogram, Chroma STFT, Spectral Contrast, Tonnetz, and Zero Crossing Rate (all features have 216 timesteps). We also tried concatenating all features together. The details of each feature are as follows:

- MFCCs (40 features): MFCCs represent the spectral envelope of the audio signal and are widely used in speech and audio processing tasks. They capture timbral and spectral characteristics of the audio.
- Mel spectrogram (32 features): The Mel spectrogram is a time-frequency representation of the audio signal that uses the Mel scale, which is more perceptually relevant than the linear frequency scale. The Mel spectrogram provides information about the spectral content and temporal changes in the audio signal.
- Chroma features (12 features): Chroma features represent the energy distribution across different pitches or frequency bands in the audio signal. They capture the harmonic and melodic content of the audio, which can be useful in various music information retrieval tasks and audio classification tasks.
- Spectral contrast (7 features): Spectral contrast measures the difference in amplitude between peaks and valleys in the audio signal's frequency spectrum. It provides information about the spectral shape and texture of the sound, which can help differentiate between different types of sounds in audio classification tasks.
- Tonnetz (6 features): Tonnetz features capture the harmonic relations between the pitches in the audio signal, providing information about the harmonic structure and musical content of the sound. They are derived from the chroma features and are useful in audio classification tasks.
- Zero-crossing rate (1 feature): Zero-crossing rate is the rate at which the audio signal changes its sign (crosses the zero-amplitude line). It is a simple feature that can provide information about the audio signal's frequency content and can be useful in tasks like speech/music classification and onset detection.

The results for each feature are presented below. The feature with the highest performance in the minority classes is the Mel spectrogram, which is a scaled version of the STFT. Consequently, we decided to simplify insights by identifying the frequency bands that can be used to predict heart conditions with reasonable efficacy.

# Results from features experimentation
| Feature used | Accuracy | Precision (murmur) | Recall (murmur) | F1-score (murmur) | Precision (extrasystole) | Recall (extrasystole) | F1-score (extrasystole) |
| - | - | - | - | - | - | - | - |
| Mel spectrogram | 0.82 | 0.59 | 1.00 | 0.75 | 0.71 | 0.91 | 0.80 |
| MFCC | 0.69 | 0.50 | 0.74 | 0.60 | 0.45 | 0.45 | 0.45 |
| Chroma STFT | 0.63 | 0.40 | 0.63 | 0.49 | 0.75 | 0.27 | 0.40 |
| Spectral Contrast | 0.55 | 0.28 | 0.26 | 0.27 | 0.20 | 0.18 | 0.19 |
| Tonnetz | 0.45 | 0.28 | 0.84 | 0.42 | 0.64 | 0.82 | 0.72 |
| Zero Crossing Rate | 0.54 | 0.41 | 0.63 | 0.50 | 0.29 | 0.73 | 0.41 |
| All features Concatenated | 0.67 | 0.50 | 0.42 | 0.46 | 0.33 | 0.09 | 0.14 |

# Results from STFT specific band frequency
| Feature used | Accuracy | Precision (murmur) | Recall (murmur) | F1-score (murmur) | Precision (extrasystole) | Recall (extrasystole) | F1-score (extrasystole) |
| - | - | - | - | - | - | - | - |
| Mel spectrogram | 0.82 | 0.59 | 1.00 | 0.75 | 0.71 | 0.91 | 0.80 |


# Insight


# limitation
- 

# Outline of the project
- Data preparation and cleaning 
    - [ ] preprocess .wav file (T)
    - [ ] noise filtering ? (T)
    - [ ] other cleaning techniques, if any. (T)
- Exploratory data analysis/visualization 
    - [ ] time series visualization of each class (KP)
    - [ ] Fourier analysis (N)
    - [ ] Other visualization, if any. (N)
- Modeling
    - [ ] ANN on fourier freq (N)
    - [ ] LSTM on whole wavelet (KP)
    
- Reference ? 
    - [Cleaning | outlier](https://pro.arcgis.com/en/pro-app/latest/tool-reference/space-time-pattern-mining/understanding-outliers-in-time-series-analysis.htm)
    - [Cleaning | denoise-1](https://www.kaggle.com/code/residentmario/denoising-algorithms/notebook) 
    - [Cleaning | denoise-2](https://github.com/ebrahimpichka/LSM-denoise)
    - [Fourier](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.66.6950&rep=rep1&type=pdf)
    - [MFCC | feature extraction](https://www.kaggle.com/code/gopidurgaprasad/mfcc-feature-extraction-from-audio/notebook)
    - [Signal Filtering](https://swharden.com/blog/2020-09-23-signal-filtering-in-python/)
    - [Classification Using Deep Learning](https://www.mdpi.com/1424-8220/19/21/4819)
