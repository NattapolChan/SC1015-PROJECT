# SC1015-PROJECT-T25
- Chanpaisit Nattapol
- Maison Sapol
- Saeng-nil Natthakan

Time Series Projects

# About
In this project, we explore [heart beat sounds dataset](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds) for beat classification.

# Preprocess

# Result
| Model | Feature used | Accuracy | TPR (murmur) | FPR (murmur) | TPR (extrastole) | FPR (extrastole) |
| - | - | - | - | - | - | - |
| ResNet10 (one vs one) | STFT | | | | | |
| ResNet10 (normal vs {murmur, extrastole}) <br/>+<br/> ResNet10 (murmur vs extrastole) | STFT | | | | | |
| | | | | | | |
# Insight
- One vs One Classification model perform much worse than One vs All model
- 

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
