import cv2
import streamlit as st
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento facial para toma de lista",
    page_icon=":smile:",
    initial_sidebar_state='auto'
)

# Función para cargar el modelo de reconocimiento facial
def cargar_modelo():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('modeloLBPHFace2.xml')
    return face_recognizer

# Tomar foto
cap = st.camera_input("Capture una foto para identificar a las personas en ella")   
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
if cap is None:
    st.text("Por favor tome una foto")
else:
    image = Image.open(cap)
    st.image(image, use_column_width=True)

    # Realizar la predicción
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        rostro = gray[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(image, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        if result[1] < 70:
            cv2.putText(image, '{}'.format(result[0]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(image, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convertir imagen de formato OpenCV a formato de imagen PIL
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Mostrar la imagen capturada
    st.image(img_pil, caption='Imagen capturada', use_column_width=True)

# Función principal
def main():
    # Cargar el modelo de reconocimiento facial
    face_recognizer = cargar_modelo()

if __name__ == '__main__':
    main()