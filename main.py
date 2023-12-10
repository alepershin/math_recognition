import streamlit as st
import cv2 as cv
import numpy as np

from tensorflow.keras.models import load_model
from PIL import Image

st.title('Проверка письменных работ по математике')

filename = st.file_uploader('Load an image', type=['png', 'jpg'])  # Добавление загрузчика файлов

if not filename is None:                       # Выполнение блока, если загружено изображение
    image = Image.open(filename)
    st.image(image)