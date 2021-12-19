import librosa
import soundfile as sf
import os
from scipy.io import wavfile
import numpy as np
import random as rn
import math
import time
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import re

# Setting seed and global variables
rn.seed(443)
NOISE_FOLDER = 'background_noise'
CLIP_FOLDER = NOISE_FOLDER+'/one_second_clips'

# Create lists used in adding noise clips to signal
CLIP_LIST_6 = [file for file in os.listdir(CLIP_FOLDER) if file.endswith('.wav') and re.search('doing_the_dishes', file)]
CLIP_LIST_5_6 = [file for file in os.listdir(CLIP_FOLDER) if file.endswith('.wav') and not re.search('doing_the_dishes', file)]
CLIP_LIST_5_6_15 = []
CLIP_LIST_5_6_85 = []

noise_names = ['running_tap', 'white_noise', 'pink_noise', 'dude_miaowing', 'exercise_bike']
for clip_file in CLIP_LIST_5_6:
    noise_name = re.split('\d', clip_file, 1)[0][:-1]
    count = [x for x in CLIP_LIST_5_6_15 if re.search(noise_name, x)]
    if len(count) >= 9:
        CLIP_LIST_5_6_85.append(clip_file)
    else:
        CLIP_LIST_5_6_15.append(clip_file)


# Padding functions used in preprocessing function
pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))
pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))

def pre_process_data(directory, noise, set_type):
    """
    Takes a directory and returns tensors of labels and data in Raw format and "MFCC format".
    Tensors are zero-padded to fit a CNN input.
    """
    
    # Dictionary used for translation of label values.
    translation = {'backward': 0, 'bed': 1, 'bird': 2,'cat': 3, 'dog': 4, 'down': 5,'eight': 6,'five': 7, 'follow': 8,
               'forward': 9, 'four': 10, 'go': 11, 'happy': 12, 'house': 13, 'learn': 14, 'left': 15, 'nine': 16,
               'no': 17, 'off': 18, 'on': 19, 'one': 20, 'right': 21, 'seven': 22, 'six': 23, 'stop': 24, 'three': 25,
               'tree': 26, 'two': 27, 'up': 28, 'visual': 29, 'wow': 30, 'yes': 31, 'zero': 32}
    
    data_raw = []
    data_mfcc = []
    labels = []
    
    file_list = sorted(os.listdir(directory))
    # Only take 50% of the samples, every second sample is kept
    file_list = file_list[::2]

    count_files = 0
    snr_ratio = [0, 20, 15, 10, 5]
    snr_idx = 0
    for file_name in file_list:
        if count_files>=35:
            pass#break
        wav, sr = librosa.load(directory + "/" + file_name)
        word_name = file_name.split('_')[0]
        # Pad wave-signal if less than one second
        wav = pad1d(wav, 22050)
        
        # If noise will be added, use clean (no noise), SNR 20, 15, 10, 5.
        # 20% in each group
        if noise:
            wav = add_noise(wav, snr_ratio[snr_idx], set_type)
            snr_idx+=1
            if snr_idx>4:
                snr_idx = 0
            
        
        # Pad and normalize data (mean 0 and std.dev 1). Only raw wave will be downsampled
        # Raw wavedata
        wav8k = librosa.resample(wav, sr, 8000)
        padded_raw = (wav8k - np.mean(wav8k)) / np.std(wav8k)
        
        # MFCC data
        mfcc = librosa.feature.mfcc(wav)
        padded_mfcc = pad2d(mfcc, 45)
        padded_mfcc = (padded_mfcc - np.mean(padded_mfcc)) / np.std(padded_mfcc)
        
        data_raw.append(padded_raw)
        data_mfcc.append(padded_mfcc)
        labels.append(translation[word_name])
        count_files+=1
        
    # Make labeling data categorical
    labels = tf.keras.utils.to_categorical(labels)
    
    # Format the data to fit tensorflow input
    data_raw = np.vstack(data_raw)
    data_raw = np.expand_dims(data_raw, -1)
    data_mfcc = np.expand_dims(np.array(data_mfcc), -1)
    
    return data_raw, data_mfcc, labels

def add_noise(wav, snr_ratio, set_type):
    """
    Takes input and fetches noise. Noise is added to the input.
    Return the fusioned input and noise.
    
    Signal-to-noise ratio:
        https://en.wikipedia.org/wiki/Signal-to-noise_ratio
        https://medium.com/analytics-vidhya/adding-noise-to-audio-clips-5d8cee24ccb8
    """
    
    # If ratio is set to 0, this means a clean signal
    if snr_ratio==0:
        return wav

    # Get 1 second noise
    if set_type in ['train_dyn_noise', 'val_noise_dyn']:
        clip_list = CLIP_LIST_5_6_85# Use 5/6 files, and only 85% of the files
    elif set_type in ['test_5_6']:
        clip_list = CLIP_LIST_5_6_15# Use 5/6 files, and only 15% of the files
    elif set_type in ['test_6']:
        clip_list = CLIP_LIST_6# Use 6th file, any part
    else:
        print("ERROR")

    file = rn.sample(clip_list, 1)
    one_sec_noise, _ = librosa.load(CLIP_FOLDER + "/" + file[0])
    wav_rms = math.sqrt(np.mean(wav**2))
    one_sec_noise_rms = math.sqrt(np.mean(one_sec_noise**2))
    
    # Use snr ratio wished for, low number gives more noise in signal
    noise_rms = wav_rms / (10**(snr_ratio/20))
    noise_factor = noise_rms / one_sec_noise_rms
    
    return wav + (noise_factor * one_sec_noise)


# Fetch input data, can choose 'small', 'medium' or 'full' size of dataset
# Split of data in train/test/validaion is done prio to this script
data_set_time_start = time.time()
print(f"Script start at {data_set_time_start}")
data_set_size = 'full'
folder_npy = 'full_repo/ex_4/'
# For training data, provide both dataset with and without noise
train_data_raw_no_noise, train_data_mfcc_no_noise, train_labels = pre_process_data(data_set_size + '_repo/train', False, 'train_no_noise')
np.save(folder_npy+'train_data_raw_no_noise', train_data_raw_no_noise)
np.save(folder_npy+'train_data_mfcc_no_noise', train_data_mfcc_no_noise)
np.save(folder_npy+'train_labels', train_labels)
print("Train data, no noise, created in", int((time.time()-data_set_time_start)/60), "minutes!")

train_data_raw_noise_dyn, train_data_mfcc_noise_dyn, train_labels = pre_process_data(data_set_size + '_repo/train', True, 'train_dyn_noise')
print("Train data, noise snr dynamic, created!", int((time.time()-data_set_time_start)/60), "minutes!")
np.save(folder_npy+'train_data_raw_noise_dyn', train_data_raw_noise_dyn)
np.save(folder_npy+'train_data_mfcc_noise_dyn', train_data_mfcc_noise_dyn)
np.save(folder_npy+'train_labels', train_labels)

test_data_raw_no_noise, test_data_mfcc_no_noise, test_labels = pre_process_data(data_set_size + '_repo/test', False, 'test_no_noise')
print("Test data no noise created!", int((time.time()-data_set_time_start)/60), "minutes!")
np.save(folder_npy+'test_data_raw_no_noise', test_data_raw_no_noise)
np.save(folder_npy+'test_data_mfcc_no_noise', test_data_mfcc_no_noise)
np.save(folder_npy+'test_labels', test_labels)

test_data_raw_5_6, test_data_mfcc_5_6, test_labels = pre_process_data(data_set_size + '_repo/test', True, 'test_5_6')
print("Test data 5 of 6 created!", int((time.time()-data_set_time_start)/60), "minutes!")
np.save(folder_npy+'test_data_raw_5_6', test_data_raw_5_6)
np.save(folder_npy+'test_data_mfcc_5_6', test_data_mfcc_5_6)
np.save(folder_npy+'test_labels', test_labels)

test_data_raw_6, test_data_mfcc_6, test_labels = pre_process_data(data_set_size + '_repo/test', True, 'test_6')
print("Test data 6th created!", int((time.time()-data_set_time_start)/60), "minutes!")
np.save(folder_npy+'test_data_raw_6', test_data_raw_6)
np.save(folder_npy+'test_data_mfcc_6', test_data_mfcc_6)
np.save(folder_npy+'test_labels', test_labels)

val_data_raw_no_noise, val_data_mfcc_no_noise, val_labels = pre_process_data(data_set_size + '_repo/validation', False, 'val_no_noise')
print("Validation data no noise created!", int((time.time()-data_set_time_start)/60), "minutes!")
np.save(folder_npy+'val_data_raw_no_noise', val_data_raw_no_noise)
np.save(folder_npy+'val_data_mfcc_no_noise', val_data_mfcc_no_noise)
np.save(folder_npy+'val_labels', val_labels)

val_data_raw_noise_dyn, val_data_mfcc_noise_dyn, val_labels = pre_process_data(data_set_size + '_repo/validation', True, 'val_noise_dyn')
print("Validation data dynamic noise created!", int((time.time()-data_set_time_start)/60), "minutes!")
np.save(folder_npy+'val_data_raw_noise_dyn', val_data_raw_noise_dyn)
np.save(folder_npy+'val_data_mfcc_noise_dyn', val_data_mfcc_noise_dyn)
np.save(folder_npy+'val_labels', val_labels)