import tkinter as tk
import customtkinter as ctk
from PIL import Image
import cv2
import threading
import mediapipe as mp
import numpy as np
import os
import time
from queue import Queue
from tensorflow.keras.models import load_model
from gtts import gTTS
import playsound
from tkinter import messagebox # <--- CAMBIO: Importado para mostrar ventanas de error

# --- 1. CONFIGURACIÓN Y PARÁMETROS ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
CARPETA_MODELO = 'modelo_entrenado'
RUTA_MODELO = os.path.join(CARPETA_MODELO, 'modelo_gestos.keras')
RUTA_ETIQUETAS = os.path.join(CARPETA_MODELO, 'etiquetas.npy')
CARPETA_AUDIOS = 'audios_mp3'
# Parámetros de MediaPipe
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
# --- PARÁMETROS DE PREDICCIÓN MEJORADOS ---
SEQUENCE_LENGTH = 30
PREDICTION_THRESHOLD = 0.85   
PREDICTION_COOLDOWN = 1.5      # Tiempo de espera entre señas reducido para mayor fluidez.
SIGN_DISPLAY_DURATION = 2.0  # Duración del texto en pantalla.

# --- 2. FUNCIONES AUXILIARES ---
if not os.path.exists(CARPETA_AUDIOS):
    os.makedirs(CARPETA_AUDIOS)
audio_queue = Queue()

def audio_worker():
    while True:
        filename = audio_queue.get()
        if filename is None: break
        try:
            playsound.playsound(filename, True)
            if os.path.exists(filename): os.remove(filename)
        except Exception as e:
            print(f"[ERROR playsound] {e}")
        finally:
            audio_queue.task_done()

def say_text(text):
    try:
        filename = os.path.join(CARPETA_AUDIOS, f"audio_{int(time.time() * 1000)}.mp3")
        tts = gTTS(text=text, lang='es', slow=False)
        tts.save(filename)
        audio_queue.put(filename)
    except Exception as e:
        print(f"[ERROR gTTS] {e}")

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def draw_styled_landmarks(image, results):
    drawing_spec_points = mp.solutions.drawing_utils.DrawingSpec(color=(100,100,100), thickness=1, circle_radius=2)
    drawing_spec_lines = mp.solutions.drawing_utils.DrawingSpec(color=(150,0,150), thickness=1, circle_radius=1)
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, drawing_spec_points, drawing_spec_lines)
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, drawing_spec_points, drawing_spec_lines)

# <--- CAMBIO: Nueva función para encontrar cámaras disponibles --->
def find_available_cameras():
    """Escanea los índices de cámara y devuelve una lista de las que están disponibles."""
    available_cameras = []
    # Revisamos los primeros 5 índices, que suele ser suficiente
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(f"Cámara {i}")
            cap.release()
    return available_cameras

# --- 3. CLASE PRINCIPAL ---
class NeuroLESSAApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NeuroLESSA - Intérprete de Señas IA")
        self.geometry("1100x720")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables de estado
        self.is_camera_on = False
        self.cap = None
        self.processing_thread = None
        self.update_id = None
        
        # Variables de lógica de señas
        self.sequence = []
        self.current_sign_text = ""
        self.last_prediction_time = 0.0
        self.sign_display_start_time = 0.0
        
        # Buffer para la imagen
        self.frame_lock = threading.Lock()
        self.current_frame = None
        
        # Carga del modelo
        try:
            self.modelo = load_model(RUTA_MODELO)
            self.actions = np.load(RUTA_ETIQUETAS)
            print("[INFO] Modelo y etiquetas cargados.")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el modelo: {e}")
            self.modelo, self.actions = None, []
            messagebox.showerror("Error de Modelo", "No se pudo cargar el modelo de IA. Asegúrate de que los archivos 'modelo_gestos.keras' y 'etiquetas.npy' estén en la carpeta 'modelo_entrenado'.")
        
        # Inicialización de MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        
        # Iniciar hilo de audio
        threading.Thread(target=audio_worker, daemon=True).start()
        
        # Crear la interfaz
        self.create_widgets()
        self.video_width, self.video_height = 1, 1

    def create_widgets(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Frame izquierdo
        self.frame_izquierdo = ctk.CTkFrame(self, width=250, corner_radius=20)
        self.frame_izquierdo.grid(row=0, column=0, padx=20, pady=20, sticky="ns")
        self.frame_izquierdo.grid_propagate(False)

        self.label_titulo = ctk.CTkLabel(self.frame_izquierdo, text="🤖 NeuroLESSA", font=ctk.CTkFont(size=28, weight="bold"))
        self.label_titulo.pack(pady=(20,10), padx=10)

        self.label_subtitulo = ctk.CTkLabel(self.frame_izquierdo, text="Intérprete de Lengua de Señas", font=ctk.CTkFont(size=14))
        self.label_subtitulo.pack(pady=(0,20), padx=10)

        # <--- CAMBIO: Selector de cámara --->
        self.camera_selection_frame = ctk.CTkFrame(self.frame_izquierdo, fg_color="transparent")
        self.camera_selection_frame.pack(pady=10, padx=20, fill="x")
        self.label_camera_select = ctk.CTkLabel(self.camera_selection_frame, text="Seleccionar Cámara:")
        self.label_camera_select.pack()
        
        self.available_cameras = find_available_cameras()
        self.selected_camera = ctk.StringVar(value=self.available_cameras[0] if self.available_cameras else "No hay cámaras")
        
        self.camera_dropdown = ctk.CTkOptionMenu(self.camera_selection_frame, variable=self.selected_camera, values=self.available_cameras, state="disabled" if not self.available_cameras else "normal")
        self.camera_dropdown.pack(pady=(5,10), fill="x")

        self.btn_toggle_camera = ctk.CTkButton(self.frame_izquierdo, text="Iniciar Cámara", command=self.toggle_camera, height=40, state="disabled" if not self.available_cameras else "normal")
        self.btn_toggle_camera.pack(pady=10, padx=20, fill="x")

        self.status_frame = ctk.CTkFrame(self.frame_izquierdo)
        self.status_frame.pack(pady=20, padx=20, fill="x")
        self.label_status_title = ctk.CTkLabel(self.status_frame, text="ÚLTIMA SEÑA DETECTADA:", font=ctk.CTkFont(size=12, weight="bold"))
        self.label_status_title.pack(pady=(10,5))
        self.label_detected_sign = ctk.CTkLabel(self.status_frame, text="---", font=ctk.CTkFont(size=36, weight="bold"), text_color="#3498db", width=200, wraplength=200, anchor="center")
        self.label_detected_sign.pack(pady=(5,15))
        
        self.btn_salir = ctk.CTkButton(self.frame_izquierdo, text="Salir", command=self.on_closing, fg_color="#D32F2F", hover_color="#B71C1C", height=40)
        self.btn_salir.pack(side="bottom", pady=20, padx=20, fill="x")

        # Frame derecho
        self.frame_derecho = ctk.CTkFrame(self, corner_radius=20)
        self.frame_derecho.grid(row=0, column=1, padx=(0,20), pady=20, sticky="nsew")
        self.label_video = ctk.CTkLabel(self.frame_derecho, text="Cámara Apagada", font=ctk.CTkFont(size=16))
        self.label_video.pack(expand=True, fill="both", padx=10, pady=10)
        self.label_video.bind("<Configure>", self.on_video_resize)

    def on_video_resize(self, event):
        self.video_width, self.video_height = event.width, event.height

    def toggle_camera(self):
        if self.is_camera_on: self.stop_camera()
        else: self.start_camera()

    def start_camera(self):
        # <--- CAMBIO: Obtener el índice de la cámara seleccionada --->
        try:
            selected_cam_str = self.selected_camera.get()
            if "Cámara" not in selected_cam_str:
                messagebox.showerror("Error", "No hay cámaras disponibles para iniciar.")
                return
            camera_index = int(selected_cam_str.split()[-1])
        except (ValueError, IndexError):
            messagebox.showerror("Error", "La selección de cámara no es válida.")
            return

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) # Usar el índice seleccionado
        
        if not self.cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara {camera_index}")
            # <--- CAMBIO: Mostrar error en la GUI --->
            messagebox.showerror("Error de Cámara", f"No se pudo abrir la '{selected_cam_str}'.\nPuede que esté en uso por otra aplicación o desconectada.")
            self.cap = None
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.is_camera_on = True
        self.sequence.clear()
        
        self.last_prediction_time = 0.0
        self.current_sign_text = ""
        
        self.processing_thread = threading.Thread(target=self.video_processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.btn_toggle_camera.configure(text="Detener Cámara")
        self.camera_dropdown.configure(state="disabled") # Deshabilitar selector mientras la cámara está activa
        self.update_gui_loop()

    def stop_camera(self):
        self.is_camera_on = False
        if self.update_id:
            self.after_cancel(self.update_id)
            self.update_id = None
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0) 
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        frame_black = Image.new('RGB', (self.video_width, self.video_height), (30,30,30))
        ctk_img = ctk.CTkImage(light_image=frame_black, dark_image=frame_black, size=(self.video_width, self.video_height))
        self.label_video.configure(image=ctk_img, text="Cámara Apagada")
        self.label_detected_sign.configure(text="---")
        self.btn_toggle_camera.configure(text="Iniciar Cámara")
        self.camera_dropdown.configure(state="normal") # <--- CAMBIO: Habilitar selector de nuevo

    def update_gui_loop(self):
        if not self.is_camera_on: return
        
        with self.frame_lock:
            img_pil = self.current_frame
            
        if img_pil:
            img_resized = img_pil.resize((self.video_width, self.video_height), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(self.video_width, self.video_height))
            self.label_video.configure(image=ctk_img, text="")
            
        if self.current_sign_text and (time.time() - self.sign_display_start_time > SIGN_DISPLAY_DURATION):
            self.current_sign_text = ""
            self.label_detected_sign.configure(text="---")
            
        self.update_id = self.after(20, self.update_gui_loop)

    def video_processing_loop(self):
        while self.is_camera_on and self.cap:
            ret, frame = self.cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            
            draw_styled_landmarks(frame, results)
            
            self.sequence.append(extract_keypoints(results))
            self.sequence = self.sequence[-SEQUENCE_LENGTH:]
            
            if len(self.sequence) == SEQUENCE_LENGTH and self.modelo:
                res = self.modelo.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                
                if np.max(res) > PREDICTION_THRESHOLD:
                    predicted_sign = self.actions[np.argmax(res)]
                    current_time = time.time()
                    
                    if current_time - self.last_prediction_time > PREDICTION_COOLDOWN:
                        if predicted_sign != self.current_sign_text:
                            # VACIAR LA COLA DE AUDIO
                            while not audio_queue.empty():
                                try:
                                    audio_queue.get_nowait()
                                except audio_queue.Empty:
                                    continue
                                audio_queue.task_done()
                            
                            # Actualizamos el estado y la GUI
                            self.current_sign_text = predicted_sign
                            self.sign_display_start_time = current_time
                            self.label_detected_sign.configure(text=self.current_sign_text.upper())
                            threading.Thread(target=say_text, args=(self.current_sign_text,), daemon=True).start()
                            
                        self.last_prediction_time = current_time
            
            # Actualizar el frame para la GUI
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with self.frame_lock:
                self.current_frame = img_pil

    def on_closing(self):
        print("Cerrando aplicación...")
        self.is_camera_on = False
        if self.update_id:
            self.after_cancel(self.update_id)
            self.update_id = None
        self.after(50, self._final_shutdown)

    def _final_shutdown(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=0.1)
        if self.cap:
            self.cap.release()
        audio_queue.put(None)
        self.holistic.close()
        self.destroy()

# --- EJECUTAR APP ---
if __name__ == "__main__":
    app = NeuroLESSAApp()
    app.mainloop()
