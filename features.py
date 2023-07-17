import numpy as np
import librosa
import os

DATA_AUDIO_FILES = os.path.join(os.getcwd(),"dataset","dataset") + "/"
audio_files_list = os.listdir(DATA_AUDIO_FILES)


emission_context = {'F' : [], 'I' : [], 'B' : []}

for file in audio_files_list:
    split = file.split('_')
    if split[0] in emission_context.keys():
        emission_context.get(split[0]).append(file)

wait_food_ex = DATA_AUDIO_FILES + emission_context.get('F')[2]   #---->'/home/fll_data_bata/DB Gatos/dataset/dataset/F_IND01_EU_FN_ELI01_301.wav'
data_food , sample_rate_F = librosa.load(wait_food_ex)
librosa.load(wait_food_ex)

isolated_ex = DATA_AUDIO_FILES+ emission_context.get('I')[0]
data_isolated , sample_rate_I  = librosa.load(isolated_ex)

brushing_ex = DATA_AUDIO_FILES + emission_context.get('B')[1]
data_brushing , sample_rate_B  = librosa.load(brushing_ex)
librosa.load(brushing_ex)



def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate_F).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate_F).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate_F).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    return result


def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate_I).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate_I).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate_I).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    return result


def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate_B).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate_B).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate_B).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    return result


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    #noise_data = noise(data)
    #res2 = extract_features(noise_data)
    #result = np.vstack((result, res2)) # stacking vertically

    # data with stretching and pitching
    #new_data = stretch(data)
    #data_stretch_pitch = pitch(new_data, sr=sample_rate)
    #res3 = extract_features(data_stretch_pitch)
    #result = np.vstack((result, res3)) # stacking vertically

    return result
