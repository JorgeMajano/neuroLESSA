# =============================================================================
# SCRIPT UNIFICADO Y MEJORADO PARA ENTRENAMIENTO DE IA DE GESTOS
# =============================================================================

import numpy as np
import tensorflow as tf
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# --- 1. CONFIGURACIÓN ---
print("[INFO] Configurando parámetros...")
# Ruta a la carpeta que contiene las carpetas de cada gesto (ej: data/hola, data/adios)
DATA_PATH = 'data'
# Parámetros de las secuencias
FRAMES_POR_SECUENCIA = 30  # El número de frames que tendrá cada secuencia (se rellenará si es menor)
# Parámetros de los landmarks (puntos clave)
# Asumiendo MediaPipe Hands: 21 landmarks con 3 coordenadas (x, y, z)
NUM_LANDMARKS = 21
DIMENSIONES_MANO = NUM_LANDMARKS * 3  # 63 por mano

# La dimensión esperada para los datos de entrada (2 manos)
# El código manejará automáticamente si solo se detecta una mano.
DIMENSION_ESPERADA = DIMENSIONES_MANO * 2  # 126 por frame

# --- 2. CARGAR Y PREPROCESAR DATOS ---
print(f"[INFO] Cargando datos desde la carpeta: '{DATA_PATH}'")

sequences, labels = [], []

# Descubrir automáticamente las acciones (gestos) a partir de los nombres de las carpetas
try:
    actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    print(f"[INFO] Gestos encontrados: {actions}")
except FileNotFoundError:
    print(f"[ERROR] La carpeta '{DATA_PATH}' no fue encontrada. Asegúrate de que la ruta es correcta.")
    exit()

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    sequence_paths = glob.glob(os.path.join(action_path, "*")) # Obtener todas las carpetas de secuencias

    for sequence_path in sequence_paths:
        # Cargar todos los frames de una secuencia
        frames_data = []
        frame_files = sorted(glob.glob(os.path.join(sequence_path, "*.npy")))
        
        for frame_file in frame_files:
            res = np.load(frame_file)
            frames_data.append(res)
        
        if not frames_data:
            print(f"[AVISO] Carpeta de secuencia vacía, ignorando: {sequence_path}")
            continue

        secuencia = np.array(frames_data)

        # --- Normalización y Relleno (Robustez del Script 1) ---
        
        # 1. Rellenar frames faltantes en la secuencia
        if len(secuencia) < FRAMES_POR_SECUENCIA:
            pad_width = FRAMES_POR_SECUENCIA - len(secuencia)
            # Rellena con ceros al final de la secuencia
            padding = np.zeros((pad_width, secuencia.shape[1]))
            secuencia = np.vstack([secuencia, padding])
        else:
            # Truncar si la secuencia es más larga
            secuencia = secuencia[:FRAMES_POR_SECUENCIA]

        # 2. Rellenar datos de la segunda mano si faltan
        if secuencia.shape[1] == DIMENSIONES_MANO:  # Solo se detectó una mano
            # Rellena con ceros para simular la segunda mano
            padding_mano = np.zeros((FRAMES_POR_SECUENCIA, DIMENSIONES_MANO))
            secuencia = np.hstack([secuencia, padding_mano])
        elif secuencia.shape[1] != DIMENSION_ESPERADA:
            print(f"[AVISO] Secuencia con formato inesperado ignorada: {sequence_path} -> {secuencia.shape}")
            continue
        
        sequences.append(secuencia)
        labels.append(action)

if not sequences:
    print("[ERROR] No se cargaron datos. Verifica que la estructura de carpetas y los archivos .npy sean correctos.")
    exit()

X = np.array(sequences, dtype=np.float32)
y = np.array(labels)

print(f"\n[INFO] Datos cargados y procesados.")
print(f"Total de ejemplos: {len(X)}")
print(f"Forma de los datos (X): {X.shape}") # (num_muestras, frames, dimensiones)
print(f"Etiquetas únicas: {np.unique(y)}")

# --- 3. CODIFICAR ETIQUETAS Y DIVIDIR DATOS ---
print("\n[INFO] Codificando etiquetas y dividiendo los datos...")

# Codificar etiquetas de texto a números (ej: 'hola' -> 0)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Convertir a formato One-Hot (ej: 0 -> [1, 0, 0])
y_categorical = to_categorical(y_encoded)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.20, random_state=42, stratify=y_categorical)

print(f"Datos de entrenamiento: {X_train.shape}")
print(f"Datos de prueba: {X_test.shape}")

# --- 4. CONSTRUIR EL MODELO LSTM ---
print("\n[INFO] Construyendo el modelo LSTM...")
num_classes = len(actions)

modelo = Sequential([
    # Capa de entrada. Define la forma de una sola muestra.
    Input(shape=(FRAMES_POR_SECUENCIA, DIMENSION_ESPERADA)),
    
    # Capa de enmascaramiento. Ignora los pasos de tiempo (frames) que son todo ceros.
    # ¡Esto es clave para manejar secuencias de diferente longitud!
    Masking(mask_value=0.0),
    
    # Capas LSTM. 'return_sequences=True' pasa la secuencia completa a la siguiente capa LSTM.
    LSTM(64, return_sequences=True, activation='tanh'),
    LSTM(128, return_sequences=True, activation='tanh'),
    # La última capa LSTM no retorna secuencias, sino un vector resumen de toda la secuencia.
    LSTM(64, return_sequences=False, activation='tanh'),
    
    # Capas densas para la clasificación final.
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax') # Softmax para clasificación multiclase
])

modelo.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

modelo.summary()

# --- 5. ENTRENAR EL MODELO ---
print("\n[INFO] Iniciando el entrenamiento...")

# Callbacks para un mejor entrenamiento:
# EarlyStopping: Detiene el entrenamiento si el 'val_loss' no mejora después de un número de épocas.
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
# ReduceLROnPlateau: Reduce la tasa de aprendizaje si el entrenamiento se estanca.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

# Aumentamos las épocas, pero EarlyStopping se encargará de detenerlo en el momento óptimo.
history = modelo.fit(
    X_train, y_train, 
    epochs=200, 
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# --- 6. GUARDAR EL MODELO Y LAS ETIQUETAS ---
print("\n[INFO] Entrenamiento completado. Guardando artefactos...")

# Nombre de la carpeta donde se guardará todo
CARPETA_MODELO = 'modelo_entrenado'
os.makedirs(CARPETA_MODELO, exist_ok=True)

# Rutas completas para los archivos
ruta_modelo = os.path.join(CARPETA_MODELO, "modelo_gestos.h5")
ruta_etiquetas = os.path.join(CARPETA_MODELO, "etiquetas_gestos.npy")

# Guardar el modelo y las etiquetas en las rutas especificadas
modelo.save(ruta_modelo)
print(f"[✅] Modelo guardado en: '{ruta_modelo}'")

np.save(ruta_etiquetas, encoder.classes_)
print(f"[✅] Etiquetas guardadas en: '{ruta_etiquetas}'")

print("\n¡Proceso finalizado con éxito!")