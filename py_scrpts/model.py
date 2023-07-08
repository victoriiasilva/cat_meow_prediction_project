import os
import IPython.display as ipd
import librosa
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras import datasets, layers, models
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split

'''ROOT ACCESS TO AUDIOS LOCALLY'''

DATA_DIR = '/Users/martinaaguilar/code/victoriiasilva/cat_meow_prediction_project/audios/dataset/dataset/'
audio_files = os.listdir(DATA_DIR)


'''MAKE DATASET BASED ON CAT'S MEOW EMISSION CONTEXT'''

# 'F' : 'Waiting For Food', 'I' : 'Isolated in unfamiliar Environment', 'B' : 'Brushing'
emission_context = {'F' : [], 'I' : [], 'B' : []}

for file in audio_files:
    split = file.split('_')
    if split[0] in emission_context.keys():
        emission_context.get(split[0]).append(file)

wait_food_ex = DATA_DIR + emission_context.get('F')[2]
data_food , sample_rate_F = librosa.load(wait_food_ex)

isolated_ex = DATA_DIR+ emission_context.get('I')[0]
data_isolated , sample_rate_I  = librosa.load(isolated_ex)

brushing_ex = DATA_DIR + emission_context.get('B')[1]
data_brushing , sample_rate_B  = librosa.load(brushing_ex)

# rename keys of dictionary
emission_context['Waiting For Food'] = emission_context.pop('F')
emission_context['Isolated in unfamiliar Environment'] = emission_context.pop('I')
emission_context['Brushing'] = emission_context.pop('B')


'''MAKE ALL DATA INTO ONE DATAFRAME'''

def add_path(list_emission):
    path = []
    for i in range(len(list_emission)):
        path.append(DATA_DIR+list_emission[i])
    return path


a = add_path(emission_context['Waiting For Food'])
b = add_path(emission_context['Isolated in unfamiliar Environment'])
c = add_path(emission_context['Brushing'])
emission = a+b+c

df = pd.DataFrame(emission, columns=['Emission'])

#a ---> path audios Food
#b ---> path audios Isolated
#c ---> path audios Brush

#emission ---> a+b+c


'''ADDING LABEL TO THE DATAFRAME'''

label_W = np.tile('Waiting For Food',92) #---> array "Waiting For Food" * 92 veces
label_I = np.tile('Isolated in unfamiliar Environment',221) #---> array "Isolated in unfamiliar Enviroment" * 221 veces
label_B = np.tile('Brushing',127) #---> array "Brushing" * 127 veces

Label = np.append(label_W, np.append(label_I, label_B))
df['Label'] = Label


'''ESPECTRO TRANSFORM (adding arrays)'''

def espectro_transform(audio_wav):
    audio_array = []

    for i in audio_wav:
        try:
            x, sr = librosa.load(i, sr=44100)
            audio_array.append(x)
        except (FileNotFoundError, librosa.LibrosaError):
            print(f"Error al cargar el archivo: {i}. Ignorando archivo.")

    return audio_array

df['Data Arrays'] = espectro_transform(df["Emission"])


'''DATA AUGMENTATION'''

augmenter = Compose([    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])
augmented_audio_list = []
for audio_array in df['Data Arrays']:
    augmented_audio = augmenter(samples=audio_array, sample_rate=44100)
    augmented_audio_list.append(augmented_audio)

df['Augmented_Data_Arrays'] = augmented_audio_list

'''MAKE A DATAFRAME WITH AUGMENTED AUDIOS'''

df_augmented = pd.DataFrame({"Emission":df["Emission"],
               "Label":df["Label"],
               "Data Arrays":df["Augmented_Data_Arrays"],
               "Augmented_Data_Arrays":"Null"
                            })

df_concat = pd.concat([df,df_augmented],ignore_index=True)
df_concat.drop("Augmented_Data_Arrays",axis=1,inplace=True)


'''SECOND PART OF MAKE A DATAFRAME WITH AUGMENTED AUDIOS'''

augmented_audio_list_2 = []

for audio_array in df_concat['Data Arrays']:
    augmented_audio = augmenter(samples=audio_array, sample_rate=44100)
    augmented_audio_list_2.append(augmented_audio)

df_concat['Augmented_Data_Arrays'] = augmented_audio_list_2

'''DATAFRAME DEFINITIVO'''

df_augmented_2 = pd.DataFrame({"Emission":df_concat["Emission"],
               "Label":df_concat["Label"],
               "Data Arrays":df_concat["Augmented_Data_Arrays"],
               "Augmented_Data_Arrays":"Null"
                            })

df_concat_OK = pd.concat([df_concat,df_augmented_2],ignore_index=True)
df_concat_OK.drop("Augmented_Data_Arrays",axis=1,inplace=True)


'''EXTRACTING FEATURES'''

#FOOD
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

#ISOLATED
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

#BRUSING
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

'''GET FEATURES'''

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

'''MAKING OUR DATA COMPATIBLE TO THE MODEL'''

x = []
y = []
for i in range(len(df_concat_OK)):
    feature = get_features(df_concat_OK['Emission'].iloc[i])
    x.append(feature)  # Append the extracted feature array directly

    label = df_concat_OK['Label'].iloc[i]
    y.append(label)

#Convertir lista a np.array
x = np.vstack(x)
y = np.array(y)

le = LabelEncoder()
y = utils.to_categorical(le.fit_transform(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2)

#Agregar una dimension extra ---> 1
X_train = np.expand_dims(x_train, axis=2)
X_test = np.expand_dims(x_test, axis=2)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

'''BASELINE MODEL'''
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.05, patience=3)
model = models.Sequential()
model.add(layers.Conv1D(64, 2, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(layers.MaxPooling1D((1)))
model.add(layers.Conv1D(256, 2, activation='relu'))
model.add(layers.MaxPooling1D((1)))
model.add(layers.Conv1D(256, 2, activation='relu'))
model.add(layers.MaxPooling1D((1)))
model.add(layers.Conv1D(1024, 2, activation='relu'))
model.add(layers.MaxPooling1D((1)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='RMSprop',
             metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=24,
                    batch_size=110,
                    validation_data=(X_test, y_test))
