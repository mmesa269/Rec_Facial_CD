from PIL import Image
import os

def resize_images_in_folder(input_folder, output_folder, size=(150, 150)):
    # Asegúrate de que la carpeta de salida exista, si no, créala
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Itera sobre los archivos en la carpeta de entrada
    for filename in os.listdir(input_folder):
        # Verifica si el archivo es una imagen
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Abre la imagen
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)
            
            # Cambia el tamaño de la imagen
            img_resized = img.resize(size)
            
            # Guarda la imagen en la carpeta de salida
            output_path = os.path.join(output_folder, filename)
            img_resized.save(output_path)

# Rutas de las carpetas de entrada y salida
input_folder_path = r'C:\Users\57300\Documents\GitHub\Rec_Facial_CD\rostros\Carlos Rueda'
output_folder_path = r'C:\Users\57300\Documents\GitHub\Rec_Facial_CD\Neo_rostros\SofiaHiguera\Carlos Rueda'

# Llama a la función para redimensionar todas las imágenes en la carpeta de entrada y guardarlas en la carpeta de salida
resize_images_in_folder(input_folder_path, output_folder_path)
