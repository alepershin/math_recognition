import streamlit as st
import cv2 as cv
import numpy as np
import os
import shutil
import base64

from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
from pathlib import Path

st.title('Проверка письменных работ по математике')

filename = st.file_uploader('Load an image', type=['jpg'])  # Добавление загрузчика файлов

if not Path("dataset").exists():
  os.mkdir("dataset")
else:
  shutil.rmtree("dataset")
  os.mkdir("dataset")

for i in range(54):
    if not Path(f"dataset/{i}").exists():
        os.mkdir(f"dataset/{i}")

if not Path("dataset/rect_56_28").exists():
  os.mkdir("dataset/rect_56_28")

if not Path("dataset/rect_112_28").exists():
  os.mkdir("dataset/rect_112_28")

for i in range(6):
    if not Path(f"dataset/rect_56_28/{i}").exists():
        os.mkdir(f"dataset/rect_56_28/{i}")
    if not Path(f"dataset/rect_112_28/{i}").exists():
        os.mkdir(f"dataset/rect_112_28/{i}")

if not filename is None:                       # Выполнение блока, если загружено изображение
    image = Image.open(filename)
    st.image(image)
    enhancer = ImageEnhance.Contrast(image)
    image_new = enhancer.enhance(2)
    st.image(image_new)
    image_new = image_new.save("img.jpg")
    im = cv.imread("img.jpg")

    # Переводим изображение в оттенки серого
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # Инвертируем цвета (черное становится белым и наоборот)
    imgray = cv.bitwise_not(imgray)

    # Находим контуры
    ret, thresh = cv.threshold(imgray, 120, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Загрузим модель для распознавания цифр и букв латинского алфавита
    model = load_model('model_28_28.keras')
    model_56_28 = load_model('model_56_28_new.keras')
    model_112_28 = load_model('model_112_28.keras')

    # Размеры картинки для распознавания
    IMG_WIDTH, IMG_HEIGHT = 28, 28

    j = 0

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w < IMG_WIDTH and h < IMG_HEIGHT:
            continue

        image = thresh[y:y + h, x:x + w]

        # Определение новых размеров холста
        new_size = max(image.shape[0], image.shape[1])

        if w < 2 * h:

            # Определение сдвига по горизонтали и вертикали
            diff_width = new_size - image.shape[1]
            diff_height = new_size - image.shape[0]

            # Расширение изображения
            top = int(diff_height / 2)
            bottom = diff_height - top
            left = int(diff_width / 2)
            right = diff_width - left
            expanded_image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT)

            image = cv.resize(expanded_image, (28, 28))

            img_np = np.array(image)                # Перевод в numpy-массив

            # Преобразование x_train в тип float32 (числа с плавающей точкой) и нормализация
            img_np = img_np.astype('float32') / 255.

            digit = np.expand_dims(img_np, axis=0)

            # Распознавание примера
            prediction = model.predict(digit)

            if max(prediction[0]) < 0.5:
                continue

            # Получение и вывод индекса самого большого элемента (это номер распознанного символа)
            pred = np.argmax(prediction[0])

            if pred <= 9:
                cv.putText(im, str(pred), (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 200, 255), 2)
            elif pred == 33:
                cv.putText(im, "x", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)
            elif pred == 47:
                cv.putText(im, "+", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)
            elif pred == 48:
                cv.putText(im, "-", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)
            elif pred == 50:
                cv.putText(im, "(", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)
            elif pred == 51:
                cv.putText(im, ")", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)
            elif pred == 36:
                cv.putText(im, "a", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
            elif pred == 34:
                cv.putText(im, "y", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
            elif pred == 23:
                cv.putText(im, "N", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
            elif pred == 19:
                cv.putText(im, "J", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
            elif pred == 36:
                cv.putText(im, "a", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
            elif pred == 49:
                cv.putText(im, "sqrt", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
            elif pred == 48:
                cv.putText(im, "Answer", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 0), 2)

            cv.rectangle(im, (x, y), (x + w, y + h), (200, 255, 200), 3)

            j += 1

            image = Image.fromarray(image)
            image = image.save(f"dataset/{pred}/new7_{j}.jpg")

        elif w < 4 * h:

            # Определение сдвига по горизонтали и вертикали
            diff_width = new_size - image.shape[1]
            diff_height = (new_size // 2) - image.shape[0]

            # Расширение изображения
            top = int(diff_height / 2)
            bottom = diff_height - top
            left = int(diff_width / 2)
            right = diff_width - left
            expanded_image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT)

            image = cv.resize(expanded_image, (56, 28))

            img_np = np.array(image)                # Перевод в numpy-массив

            # Преобразование x_train в тип float32 (числа с плавающей точкой) и нормализация
            img_np = img_np.astype('float32') / 255.

            digit = np.expand_dims(img_np, axis=0)

            # Распознавание примера
            prediction = model_56_28.predict(digit)

            if max(prediction[0]) < 0.5:
                continue

            # Получение и вывод индекса самого большого элемента (это номер распознанного символа)
            pred = np.argmax(prediction[0])

            if pred == 0:
                cv.putText(im, "-", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
            if pred == 1:
                cv.putText(im, "sqrt", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)
            if pred == 2:
                cv.putText(im, "Answer", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 255), 2)
            if pred == 3:
                cv.putText(im, "No", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 255), 2)
            if pred == 4:
                cv.putText(im, "solution", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 255), 2)
            if pred == 5:
                cv.putText(im, "check", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 255), 2)

            cv.rectangle(im, (x, y), (x + w, y + h), (200, 255, 200), 3)

            j += 1

            image = Image.fromarray(image)
            image = image.save(f"dataset/rect_56_28/{pred}/new31_{j}.jpg")

        else:

            # Определение сдвига по горизонтали и вертикали
            diff_width = new_size - image.shape[1]
            diff_height = (new_size // 2) - image.shape[0]

            # Расширение изображения
            top = int(diff_height / 2)
            bottom = diff_height - top
            left = int(diff_width / 2)
            right = diff_width - left
            expanded_image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT)

            image = cv.resize(expanded_image, (112, 28))

            img_np = np.array(image)                # Перевод в numpy-массив

            # Преобразование x_train в тип float32 (числа с плавающей точкой) и нормализация
            img_np = img_np.astype('float32') / 255.

            digit = np.expand_dims(img_np, axis=0)

            # Распознавание примера
            prediction = model_112_28.predict(digit)

            if max(prediction[0]) < 0.5:
                continue

            # Получение и вывод индекса самого большого элемента (это номер распознанного символа)
            pred = np.argmax(prediction[0])

            if pred == 0:
                cv.putText(im, "-", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
            if pred == 1:
                cv.putText(im, "sqrt", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)

            cv.rectangle(im, (x, y), (x + w, y + h), (200, 255, 200), 3)

            j += 1

            image = Image.fromarray(image)
            image = image.save(f"dataset/rect_112_28/{pred}/new7_{j}.jpg")

    st.image(im)

    shutil.make_archive("dataset", 'zip', "dataset")

    with open("dataset.zip", 'rb') as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download=\'dataset.zip\'>\
                download file \
            </a>'
        st.markdown(href, unsafe_allow_html=True)
