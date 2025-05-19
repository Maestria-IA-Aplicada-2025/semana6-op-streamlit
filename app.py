
import streamlit as st
import cv2
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from PIL import Image

# Cargar el conjunto de datos de dígitos
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Crear y entrenar el clasificador (SVM)
pca = PCA(n_components=40)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train_pca, y_train)

# Título de la aplicación
st.title("Clasificación de Imágenes de Dígitos Manuscritos")

# Subir una imagen para la predicción
uploaded_file = st.file_uploader("Sube una imagen de un dígito", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convertir la imagen subida a escala de grises
    image = Image.open(uploaded_file).convert("L")
    image = np.array(image)

    # Redimensionar la imagen a 8x8 píxeles para que coincida con el tamaño de las imágenes del conjunto de datos
    image_resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    st.image(image_resized, caption="Imagen Preprocesada", use_column_width=True)

    # Aplanar la imagen y realizar la transformación PCA
    image_flat = image_resized.flatten().reshape(1, -1)
    image_pca = pca.transform(image_flat)

    # Realizar la predicción
    prediction = clf.predict(image_pca)

    st.write(f"Predicción: {prediction[0]}")
