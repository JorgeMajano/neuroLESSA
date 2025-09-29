import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import load_model
import pyttsx3
import threading
# --- 1. CONFIGURACIÓN INICIAL Y CARGA DEL MODELO ---
# La misma carpeta que definimos en el script de entrenamiento
CARPETA_MODELO = 'modelo_entrenado'
ruta_modelo = os.path.join(CARPETA_MODELO, 'modelo_gestos.h5')
ruta_etiquetas = os.path.join(CARPETA_MODELO, 'etiquetas_gestos.npy')
# Cargar el modelo y las etiquetas dinámicamente
try:
    modelo = load_model(ruta_modelo)
    # ¡Importante! Cargamos las etiquetas desde el archivo, no las escribimos a mano.
    actions = np.load(ruta_etiquetas) 
    print(f"[INFO] Modelo y etiquetas cargados correctamente. Gestos reconocidos: {actions}")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo o las etiquetas desde la carpeta '{CARPETA_MODELO}'.")
    print(f"Detalle del error: {e}")
    exit()
# --- 2. CONFIGURACIÓN DEL MOTOR DE VOZ ---
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150) # Velocidad del habla
except Exception as e:
    print(f"Error inicializando el motor de voz: {e}")
    engine = None
# --- 3. FUNCIÓN PARA HABLAR EN UN HILO SEPARADO ---
def say_text(text_to_say):
    if engine and text_to_say:
        try:
            engine.say(text_to_say)
            engine.runAndWait()
        except Exception as e:
            print(f"Error en la función de voz: {e}")
# --- 4. FUNCIÓN DE EXTRACCIÓN DE KEYPOINTS (ACTUALIZADA Y CRÍTICA) ---
# Esta función ahora extrae los datos de AMBAS manos, que es como el modelo fue entrenado.
def extract_keypoints(results):
    # Extraer landmarks de mano izquierda y derecha, rellenando con ceros si no se detectan.
    # El tamaño de cada mano es 21 landmarks * 3 coordenadas = 63.
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    # Concatenar ambos para tener el vector de 126 que el modelo espera.
    return np.concatenate([lh, rh])
# --- 5. INICIALIZACIÓN DE MEDIAPIPE Y CÁMARA ---
cap = cv2.VideoCapture(0)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
# Variables para la lógica de detección
sequence = []
sentence = []
predictions = []
threshold = 0.8 # Confianza mínima para una predicción
last_word_said = ""
# --- 6. BUCLE PRINCIPAL DE DETECCIÓN ---
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1) # Voltear la imagen horizontalmente
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        # Dibujar landmarks
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # Lógica de predicción
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] # Mantener siempre los últimos 30 frames
        if len(sequence) == 30:
            res = modelo.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            prediction_index = np.argmax(res)
            # Lógica para una predicción más estable
            if res[prediction_index] > threshold:
                prediction_word = actions[prediction_index]
                # Si la palabra es nueva y diferente a la última detectada
                if prediction_word != last_word_said:
                    sentence.append(prediction_word)
                    last_word_said = prediction_word # Actualizar la última palabra
                    # Inicia un hilo para decir la palabra detectada.
                    threading.Thread(target=say_text, args=(prediction_word,)).start()
        if len(sentence) > 5: 
            sentence = sentence[-5:]
        # Visualización en pantalla
        cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Interprete de Señas con Voz', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()