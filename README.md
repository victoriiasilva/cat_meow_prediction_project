# cat_meow_prediction_project

This project consists of a cat-meow classificator using audios of the meow as input! We trained a model using Convolutional Neural Networks, CNN, a Deep Learning architecture. 

## process

### data

You can find the raw data here: https://www.kaggle.com/datasets/andrewmvd/cat-meow-classification

### steps:
1. We performed some exploratory analysis of the dataset and the audio. We decided which features would be relevant to the prediction
2. Using librosa library we decided to load and extract the important information about the audio and transform them into spectrograms so we could work better
3. As we need more information and data we performed Data Augmentation for the audios. We changed things like ZCR, chroma_stft, MFCC, Root Mean Square Value, and MelSpectogram so the new audios would have some little alterations and would fit as new data
4. We train the model and made the prediction in three categories: Brushing, Waiting for food, or Isolated in a different environment
5. When the model performed correctly we decided to add some tools to make the model more "user friendly" and we passed it to a Streamlit app so it could be more didactic to use and make the prediction
