import cv2
import numpy as np
import os
import mediapipe as mp
# --- 1. CONFIGURACIÓN ---
print("[INFO] Configurando parámetros...")
# Carpeta principal donde se guardarán los datos
DATA_PATH = 'data' 
# Gestos que se van a recolectar
# Puedes añadir o quitar gestos según tus necesidades
actions = np.array(['hola', 'mucho gusto',])
# Número de secuencias (videos) por cada gesto
num_secuencias = 60
# Número de frames por cada secuencia
frames_por_secuencia = 30
# --- 2. CREACIÓN DE CARPETAS ---
print(f"[INFO] Creando estructura de carpetas en '{DATA_PATH}'...")
for action in actions:
    for sequence in range(num_secuencias):
        try:
            # Crea la ruta completa: data/nombre_gesto/numero_secuencia
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except FileExistsError:
            pass # La carpeta ya existe, no hay problema
print("[INFO] Estructura de carpetas lista.")
# --- 3. INICIALIZAR MEDIAPIPE Y CÁMARA ---
print("[INFO] Inicializando MediaPipe y cámara...")
# Cargar el modelo Holistic de MediaPipe que detecta manos, cara y pose
mp_holistic = mp.solutions.holistic 
# Utilidad para dibujar los landmarks sobre la imagen
mp_drawing = mp.solutions.drawing_utils 
# Iniciar captura de video desde la cámara web (índice 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] No se pudo abrir la cámara.")
    exit()
# --- 4. LÓGICA DE RECOLECCIÓN ---
# Usar el modelo Holistic con una confianza de detección y seguimiento del 50%
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Bucle principal que se ejecuta mientras la cámara esté abierta
    # Recorrer cada gesto, luego cada secuencia de ese gesto
    for action in actions:
        for sequence in range(num_secuencias):
            # --- Preparación para la grabación de la secuencia ---
            print(f"\n[INFO] Recolectando para el gesto '{action}', secuencia número {sequence}...")
            # Mostrar un mensaje de preparación en pantalla
            for i in range(5, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] No se pudo leer el frame de la cámara.")
                    break
                # Voltear la imagen horizontalmente para un efecto espejo
                frame = cv2.flip(frame, 1)
                # Escribir texto en la imagen
                texto_preparacion = f"Preparate para '{action}'. Empezando en {i}..."
                cv2.putText(frame, texto_preparacion, (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Recoleccion de Datos', frame)
                cv2.waitKey(1000) # Esperar 1 segundo
            if not ret: break
            # --- Grabación de los frames de la secuencia ---
            for frame_num in range(frames_por_secuencia):
                # Leer un frame de la cámara
                ret, frame = cap.read()
                if not ret:
                    break
                # Voltear la imagen horizontalmente
                frame = cv2.flip(frame, 1)
                # Realizar detección con MediaPipe
                # 1. Convertir la imagen de BGR (OpenCV) a RGB (MediaPipe)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False # Optimización: marcar como no escribible
                # 2. Procesar la imagen y obtener los resultados
                results = holistic.process(image)
                # 3. Volver a convertir a BGR para mostrar con OpenCV
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # Dibujar los landmarks de las manos en la imagen
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                # --- Extracción y guardado de los puntos clave ---
                # Extraer landmarks de la mano izquierda (si existen) o crear un array de ceros
                lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
                # Extraer landmarks de la mano derecha (si existen) o crear un array de ceros
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
                # Concatenar los landmarks de ambas manos para tener un único vector de 126
                keypoints = np.concatenate([lh, rh])
                # Guardar los keypoints en un archivo .npy
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                # --- Mostrar información en pantalla ---
                info_texto = f"Gesto: {action} | Sec: {sequence} | Frame: {frame_num}"
                cv2.putText(image, info_texto, (15, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Recoleccion de Datos', image)
                # Salir del bucle si se presiona la tecla 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
# --- 5. LIBERAR RECURSOS ---
print("\n[INFO] Finalizando y liberando recursos...")
cap.release()
cv2.destroyAllWindows()
print("[INFO] Proceso completado.")