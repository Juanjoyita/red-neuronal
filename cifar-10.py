# -*- coding: utf-8 -*-
"""
CIFAR-10: Clasificación de Alta Precisión (90%+)
Incluye: Visualización, Entrenamiento Avanzado e Interfaz de Usuario
@author: Omar Y Juan
"""

# ==============================
# 1. LIBRERÍAS
# ==============================
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image

# ==============================
# 2. CARGA Y PREPROCESAMIENTO
# ==============================
print("Cargando Dataset...")
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = cifar10.load_data()

class_names = ['Avión','Auto','Pájaro','Gato','Ciervo','Perro','Rana','Caballo','Barco','Camión']

# Visualización Inicial (Muestra del Profe)
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(x_train_raw[i])
    plt.xlabel(class_names[y_train_raw[i][0]])
plt.suptitle("Muestra del Dataset Original")
plt.show()

# Preparación Técnica
y_train = to_categorical(y_train_raw, 10)
y_test = to_categorical(y_test_raw, 10)

# Normalización Z-Score
mean = np.mean(x_train_raw, axis=(0, 1, 2, 3))
std = np.std(x_train_raw, axis=(0, 1, 2, 3))
x_train = (x_train_raw - mean) / (std + 1e-7)
x_test = (x_test_raw - mean) / (std + 1e-7)

# ==============================
# 3. DATA AUGMENTATION
# ==============================
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)
datagen.fit(x_train)

# ==============================
# 4. ARQUITECTURA DEL MODELO (Mini-VGG)
# ==============================
def build_model():
    model = models.Sequential()
    wd = 1e-4 

    model.add(layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(wd), input_shape=(32,32,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(wd), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(wd), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# ==============================
# 5. ENTRENAMIENTO
# ==============================
print("\nIniciando entrenamiento...")
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

# Nota: 45 épocas es un buen número, pero para el 90%+ a veces se necesitan 80-100.
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=45, 
    validation_data=(x_test, y_test),
    callbacks=[checkpoint, lr_reducer, early_stop]
)

# ==============================
# 6. EVALUACIÓN Y PREDICCIONES (ESTILO PROFE)
# ==============================
model = models.load_model('best_model.keras')

# --- PREDICCIÓN INDIVIDUAL (La que pidió el usuario) ---
print("\n--- MOSTRANDO PREDICCIÓN DE PRUEBA (MUESTRA) ---")
img_idx = 0 
img_test_raw = x_test_raw[img_idx]
img_test_norm = x_test[img_idx]

pred_muestra = model.predict(np.expand_dims(img_test_norm, axis=0))
clase_p = class_names[np.argmax(pred_muestra)]
clase_r = class_names[np.argmax(y_test[img_idx])]

plt.figure(figsize=(5,5))
plt.imshow(img_test_raw)
plt.title(f"PREDICCIÓN: {clase_p} | REAL: {clase_r}")
plt.axis('off')
plt.show()

# Cuadrícula de Resultados
y_probs = model.predict(x_test)
y_pred = np.argmax(y_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

plt.figure(figsize=(10, 12))
for i in range(16):
    plt.subplot(4, 4, i+1)
    img_show = x_test[i] * std + mean
    img_show = np.clip(img_show, 0, 1)
    plt.imshow(img_show)
    color = 'green' if y_true[i] == y_pred[i] else 'red'
    plt.title(f"R: {class_names[y_true[i]]}\nP: {class_names[y_pred[i]]}", color=color, fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()

# ==============================
# 7. INTERFAZ PARA EL USUARIO (AL FRENTE)
# ==============================
def predecir_imagen_nueva():
    print("\n--- SELECCIÓN DE IMAGEN PERSONALIZADA ---")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True) # 🔥 ESTO HACE QUE APAREZCA ENCIMA DE TODO
    
    file_path = filedialog.askopenfilename(title="Selecciona una imagen para el modelo")
    
    if file_path:
        img_original = Image.open(file_path).convert('RGB')
        img_resized = img_original.resize((32, 32))
        img_array = np.array(img_resized)
        
        img_norm = (img_array - mean) / (std + 1e-7)
        img_tensor = np.expand_dims(img_norm, axis=0)
        
        res = model.predict(img_tensor)[0]
        idx = np.argmax(res)
        
        plt.figure(figsize=(5,4))
        plt.imshow(img_original)
        plt.title(f"Predicción: {class_names[idx]}\nConfianza: {res[idx]*100:.2f}%")
        plt.axis('off')
        plt.show()
        print(f"Resultado: Es un {class_names[idx]} con {res[idx]*100:.2f}% de seguridad.")
    else:
        print("No se seleccionó ninguna imagen.")

predecir_imagen_nueva()