from features import get_features
import tempfile
import os
import shutil
import keras
import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog
from google.cloud import storage

class Cat_Class:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    #def open_file_dialog(self):
        #root = tk.Tk()
        #root.withdraw()
        #file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav")])
        #root.destroy()
        #return file_path

    def save_audio(self,file_path):
        #file_path = self.open_file_dialog()
        if file_path:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, os.path.basename(file_path))
                shutil.copy(file_path, temp_file)
                result = self.prediction(temp_file)
            return result
        else:
            return "No file has been found"

    def prediction(self, audio_file):
        results = []
        result = ""

        try:
            features = get_features(audio_file)
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=2)

            predict = self.model.predict(features)
            result = np.argmax(predict, axis=1)
            results.append(result)

            if results[0][0] == 0:
                result = "Your cat wants to be brushed"
            elif results[0][0] == 1:
                result = "Your cat feels isolated"
            elif results[0][0] == 2:
                result = "Your cat is hungry"
        except (FileNotFoundError, librosa.LibrosaError):
            result = f"Error loading file: {audio_file}"

        return result

#Pedirme acceso al bucket


local_path = "/Users/martinaaguilar/code/victoriiasilva/cat_meow_prediction_project/modelo/trained_model.h5" #Cambiar cada uno donde va a guardar el modelo

#cat_init = Cat_Class(local_path)
#resultado = cat_init.save_audio()
#print(resultado)
