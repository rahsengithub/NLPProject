# Get word-based variables and sound based variables in same python script

#word-based, load JSON file
import json
from statistics import mean

with open('si1.json') as f:
    data = json.load(f)

# sound based, load .wav file
import scipy.io.wavfile as wav
import numpy as np
import speechpy
#import os


file_name = 'audio.wav'
#file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'audio.wav')
fs, signal = wav.read(file_name)
signal = signal[:,0]


# Extract features from JSON dicionary

# all tokens
tokens = data["tokens"]
time = data["speechLength"]*(1/60)
# get distinct words
dist_list = []
SIL_lengths = []
for tok in tokens:
    new_tok = tok["text"]
    if new_tok == "SIL":
        SIL_lengths.append(tok["duration"])
    else:
        if new_tok not in dist_list:
            dist_list.append(new_tok)
        else: continue

SIL_lengts = SIL_lengths[1:-1]
SIL_lengths = [sil for sil in SIL_lengths if sil > 2.0]
print(SIL_lengths)
pause_freq = len(SIL_lengths)/time
voc_freq = len(dist_list)/time
dist_lengths = len("hei")
 
voc_avg = mean([len(string) for string in dist_list])





# Extract features from signal using speechpy library


# Example of pre-emphasizing.
signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)

# Example of staching frames
frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01, filter=lambda x: np.ones((x,)),
         zero_padding=True)

# Example of extracting power spectrum
power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=512)

############# Extract MFCC features #############
mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)

mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)

############# Extract logenergy features #############
logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
logenergy_feature_cube = speechpy.feature.extract_derivative_feature(logenergy)

# print all features
print("Word based features for example data")
print("The frequency of pauses is: {} pauses/min".format(round(pause_freq,2)))
print("The frequency of distinct words is: {} distinct words/min".format(round(voc_freq,2)))
print("The average length of distinct words is: {} letter/word".format(round(voc_avg,2)))
print("Sound based features for example data")
print('power spectrum shape=', power_spectrum.shape)
print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)
print('mfcc feature cube shape=', mfcc_feature_cube.shape)
print('logenergy features=', logenergy.shape)

