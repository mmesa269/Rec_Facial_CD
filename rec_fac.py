import cv2
import streamlit as st
import PIL.Image
import PIL
from PIL import Image
import numpy as np
import os

dataPath = r'C:\Users\57300\Documents\GitHub\Rec_Facial_CD\rostros'
imagePaths = os.listdir(dataPath)

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento facial para toma de lista",
    page_icon=":smile:",
    initial_sidebar_state='auto'
)

# Función para cargar el modelo de reconocimiento facial
def cargar_modelo():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('modeloLBPHFace5.xml')
    return face_recognizer

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

# Tomar foto
st.subheader("Cargar Foto desde el Computador")
    
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    imagen_pil = PIL.Image.open(uploaded_file)
    cap = np.array(imagen_pil)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if cap is None:
        st.text("Por favor tome una foto")
    else:
        # Realizar la predicción
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = gray[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            face_recognizer.read('modeloLBPHFace5.xml')
            result = face_recognizer.predict(rostro)

            cv2.putText(cap, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

            if result[1] < 70:
                cv2.putText(cap, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_names.append(imagePaths[result[0]])
            else:
                cv2.putText(cap, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Convertir imagen de formato OpenCV a formato de imagen PIL
        img_pil2 = Image.fromarray((cap))

        # Mostrar la imagen capturada
        st.image(img_pil2, caption='Imagen capturada', use_column_width=True)

        st.write("Gente que ha asistido a clase hoy:")
        if not face_names:
            st.write("No fue nadie")
        else:
            for item in face_names:
                st.write(item)