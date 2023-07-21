import json
import requests
import streamlit as st
from Class import Cat_Class
from google.cloud import storage

titulo = "Cat Meow Prediction Project"
icon = ":cat:"

st.set_page_config(page_title=titulo,page_icon=icon)

with st.container():
    st.title("Cat Meow Prediction :cat:")

with st.container():
    st.write("Bienvenido al modelo de prediccion de sonidos de gatos, que te dirá qué es lo que necesita tu gato.")

with st.container():
    st.write("---")
    st.header("Nuestro model")
    st.write("##")
    upload_column, text_column = st.columns((1,2))
    with upload_column:
        file = st.file_uploader("Pone tu audio aci: ",type=["wav","mp3"])
    with text_column:
        st.subheader("Pone tu audio aqui")
        st.write("Sube un archivo de audio del meow de tu gato aquí y deja que el modelo te diga lo que necesita.")


if file is not None:
    local_path = "/Users/martinaaguilar/code/victoriiasilva/cat_meow_prediction_project/modelo/trained_model.h5"

    cat_init = Cat_Class(local_path)
    resultado = cat_init.prediction(file)
    st.write(resultado)
