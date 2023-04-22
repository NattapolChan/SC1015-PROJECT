# SC1015-PROJECT-B140-Team2
- Chanpaisit Nattapol
- Maison Sapol
- Saeng-nil Natthakan

# About
In this project, we examine the [heart beat sounds dataset](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds), with a particular focus on developing the initial screening stage for cardiac pathologies in a hospital setting. Our attention is directed towards ```dataset/set_b```, which consists of heartbeat audio files obtained from clinical trials in hospitals using digital stethoscopes. The dataset provides us with audio classified into **normal**, **murmur**, and **extrasystole** conditions. Through trial and error, we have discovered interesting findings by experimenting with features that can contribute to the aforementioned classification. Our focus is then placed on classifying heartbeat audio based on different features.

# Exploratory Data Analysis (EDA)
The number of data points in each class is determined: {normal: 320, murmur: 95, extrasystole: 46}. This indicates an imbalance in the data that needs to be addressed. We visualised the sound waveform, frequency spectrum, and power spectrogram. We identified the presence of noise in some audio files, which requires cleaning. Additionally, we noticed the variation in length of each sound data. This requires an additional step of preprocess (padding/slicing) later before training the model.
# Preprocess
We denoised audio files by eliminating specific frequencies that persist in a power spectral density noted in the visualised spectrograms. The visualisations of the Short-Time Fourier Transform (STFT) before and after denoising can be seen in ```EDA+Preprocessing.ipynb```. Additionally, we attempted to reduce the effect of imbalance of data classes by introducing class weights. Unfortunately, this approach yielded unpromising results. Instead, data augmentation by adding low intensity white noise is used for oversampling the underrepresented classes (murmur and extrasystole).
# Machine Learning models
In this project, ```InceptionTime``` architecture is utilised. We found that InceptionTime generally provides better performance while containing fewer parameters, resulting in shorter training times. As a result, all of our models are based on the InceptionTime architecture. We attempted to feed the model with raw audio data, but the results were unsatisfactory, and it took an enormous amount of time to train the model due to the excessive length of the features. As a result, we then experimented with features provided by the ```librosa``` library, such as Mel-frequency cepstral coefficients (MFCCs), Mel spectrogram, Chroma STFT, Spectral Contrast, Tonnetz, and Zero Crossing Rate (all features have 216 timesteps). We also tried concatenating all features together. The details of each feature are as follows:

- **MFCCs** (40 features): MFCCs represent the spectral envelope of the audio signal and are widely used in speech and audio processing tasks. They capture timbral and spectral characteristics of the audio.

- **Mel spectrogram** (32 features): The Mel spectrogram is a time-frequency representation of the audio signal that uses the Mel scale, which is more perceptually relevant than the linear frequency scale. The Mel spectrogram provides information about the spectral content and temporal changes in the audio signal.

- **Chroma features** (12 features): Chroma features represent the energy distribution across different pitches or frequency bands in the audio signal. They capture the harmonic and melodic content of the audio, which can be useful in various music information retrieval tasks and audio classification tasks.

- **Spectral contrast** (7 features): Spectral contrast measures the difference in amplitude between peaks and valleys in the audio signal's frequency spectrum. It provides information about the spectral shape and texture of the sound, which can help differentiate between different types of sounds in audio classification tasks.

- **Tonnetz** (6 features): Tonnetz features capture the harmonic relations between the pitches in the audio signal, providing information about the harmonic structure and musical content of the sound. They are derived from the chroma features and are useful in audio classification tasks.

- **Zero-crossing rate** (1 feature): Zero-crossing rate is the rate at which the audio signal changes its sign (crosses the zero-amplitude line). It is a simple feature that can provide information about the audio signal's frequency content and can be useful in tasks like speech/music classification and onset detection.

The results for each feature are presented below. The feature with the highest performance in the minority classes is the Mel spectrogram, which is a scaled version of the STFT. Consequently, we decided to further explore by identifying the frequency bands that can be used to predict heart conditions with reasonable efficacy instead of using the whole range of frequency.

# Results from features experimentation
<table>
<tr>
    <th rowspan="2">Feature used</th>
    <th rowspan="2">Accuracy</th>
    <th colspan="3">Normal</th>
    <th colspan="3">Murmur</th>
    <th colspan="3">Extrasystole</th>
</tr>
<tr>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
</tr>
<tr>
    <td>Mel spectrogram</td>
    <td>0.82</td>
    <td>1.00</td>
    <td>0.75</td>
    <td>0.85</td>
    <td>0.59</td>
    <td>1.00</td>
    <td>0.75</td>
    <td>0.71</td>
    <td>0.91</td>
    <td>0.80</td>
</tr>
<tr>
    <td>MFCC</td>
    <td>0.83</td>
    <td>0.71</td>
    <td>0.77</td>
    <td>0.69</td>
    <td>0.50</td>
    <td>0.74</td>
    <td>0.60</td>
    <td>0.45</td>
    <td>0.45</td>
    <td>0.45</td>
</tr>
<tr>
    <td>Chroma STFT</td>
    <td>0.75</td>
    <td>0.70</td>
    <td>0.72</td>
    <td>0.63</td>
    <td>0.40</td>
    <td>0.63</td>
    <td>0.49</td>
    <td>0.75</td>
    <td>0.27</td>
    <td>0.40</td>
</tr>
<tr>
    <td>Spectral Contrast</td>
    <td>0.68</td>
    <td>0.70</td>
    <td>0.69</td>
    <td>0.55</td>
    <td>0.28</td>
    <td>0.26</td>
    <td>0.27</td>
    <td>0.20</td>
    <td>0.18</td>
    <td>0.19</td>
</tr>
<tr>
    <td>Tonnetz</td>
    <td>0.81</td>
    <td>0.27</td>
    <td>0.40</td>
    <td>0.45</td>
    <td>0.28</td>
    <td>0.84</td>
    <td>0.42</td>
    <td>0.64</td>
    <td>0.82</td>
    <td>0.72</td>
</tr>
<tr>
    <td>Zero Crossing Rate</td>
    <td>0.83</td>
    <td>0.48</td>
    <td>0.61</td>
    <td>0.54</td>
    <td>0.41</td>
    <td>0.63</td>
    <td>0.50</td>
    <td>0.29</td>
    <td>0.73</td>
    <td>0.41</td>
</tr>
<tr>
    <td>All features Concatenated</td>
    <td>0.72</td>
    <td>0.84</td>
    <td>0.77</td>
    <td>0.67</td>
    <td>0.50</td>
    <td>0.42</td>
    <td>0.46</td>
    <td>0.33</td>
    <td>0.09</td>
    <td>0.14</td>
</tr>
</table>

# Results from STFT specific band frequency
<table>
<tr>
    <th rowspan="2">Feature used</th>
    <th rowspan="2">Accuracy</th>
    <th colspan="3">Normal</th>
    <th colspan="3">Murmur</th>
    <th colspan="3">Extrasystole</th>
</tr>
<tr>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
</tr>
<tr>
    <td>Mel spectrogram</td>
    <td>0.82</td>
    <td>1.00</td>
    <td>0.75</td>
    <td>0.85</td>
    <td>0.59</td>
    <td>1.00</td>
    <td>0.75</td>
    <td>0.71</td>
    <td>0.91</td>
    <td>0.80</td>
</tr>
<tr>
    <td>STFT (0-100 Hz)</td>
    <td>0.59</td>
    <td>0.41</td>
    <td>0.71</td>
    <td>0.52</td>
    <td>0.81</td>
    <td>0.83</td>
    <td>0.82</td>
    <td>0.57</td>
    <td>0.69</td>
    <td>0.62</td>
</tr>
<tr>
    <td>STFT (100-200 Hz)</td>
    <td>0.51</td>
    <td>0.75</td>
    <td>0.40</td>
    <td>0.52</td>
    <td>0.47</td>
    <td>0.44</td>
    <td>0.46</td>
    <td>0.39</td>
    <td>0.79</td>
    <td>0.53</td>
</tr>
<tr>
    <td>STFT (200-300 Hz)</td>
    <td>0.61</td>
    <td>0.87</td>
    <td>0.28</td>
    <td>0.42</td>
    <td>0.69</td>
    <td>0.63</td>
    <td>0.66</td>
    <td>0.53</td>
    <td>1.00</td>
    <td>0.69</td>
</tr>
<tr>
    <td>STFT (300-400 Hz)</td>
    <td>0.57</td>
    <td>0.60</td>
    <td>0.34</td>
    <td>0.43</td>
    <td>0.80</td>
    <td>0.63</td>
    <td>0.71</td>
    <td>0.41</td>
    <td>0.87</td>
    <td>0.56</td>
</tr>
<tr> 
    <td>STFT (400-500 Hz)</td>
    <td>0.60</td>
    <td>0.92</td>
    <td>0.29</td>
    <td>0.44</td>
    <td>0.66</td>
    <td>0.72</td>
    <td>0.69</td>
    <td>0.48</td>
    <td>0.89</td>
    <td>0.63</td>
</tr>
<tr>
    <td>STFT (500-600 Hz)</td>
    <td>0.39</td>
    <td>0.75</td>
    <td>0.27</td>
    <td>0.39</td>
    <td>0.30</td>
    <td>0.36</td>
    <td>0.33</td>
    <td>0.34</td>
    <td>0.60</td>
    <td>0.43</td>
</tr>
<tr>
    <td>STFT (600-700 Hz)</td>
    <td>0.47</td>
    <td>0.56</td>
    <td>0.53</td>
    <td>0.55</td>
    <td>0.44</td>
    <td>0.29</td>
    <td>0.35</td>
    <td>0.37</td>
    <td>0.47</td>
    <td>0.42</td>
</tr>
<tr>
    <td>STFT (700-800 Hz)</td>
    <td>0.20</td>
    <td>0.84</td>
    <td>0.17</td>
    <td>0.29</td>
    <td>0.00</td>
    <td>0.00</td>
    <td>0.00</td>
    <td>0.12</td>
    <td>1.00</td>
    <td>0.22</td>
</tr>
</table>


**NOTE:** Features that exhibit practical performance should be able to detect the diseased classes (murmur and extrasystole) as part of the feasible screening stage, i.e., erroneously classifying *normal* as *diseased* is not as lethal as classifying *diseased* as *normal*. The goal in this case is then to maximise the recall and precision of minority classes.

# Insight


# Limitation
- Dataset is highly imbalance in class (320 for 'normal', compared to 45, 96 for 'extrasystole' and 'murmur' respectively)
- Number of data is considerably small
- Detecting and removing background noise is challenging and can be inaccurate.
- 
    
# Reference
- [Signal Filtering](https://swharden.com/blog/2020-09-23-signal-filtering-in-python/)
- [Audio Data Augmentation](https://pytorch.org/audio/main/tutorials/audio_data_augmentation_tutorial.html#)
- [Fourier](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.66.6950&rep=rep1&type=pdf)
- [Mel Spectrogram](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)
- [MFCC](https://medium.com/@tanveer9812/mfccs-made-easy-7ef383006040)
- [Audio Feature Extraction](https://librosa.org/doc/main/feature.html)
- [InceptionTime](https://link.springer.com/article/10.1007/s10618-020-00710-y)
