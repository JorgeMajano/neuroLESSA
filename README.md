# 🤖 neuroLESSA

**Inteligencia Artificial para la Interpretación en tiempo real de la Lengua de Señas Salvadoreña (LESSA) a texto y voz.**

-----

### 📖 Acerca del Proyecto

En El Salvador, existe una persistente barrera comunicativa entre personas con discapacidad auditiva y la comunidad oyente. A pesar de la existencia de soluciones de IA, muchas no están adaptadas a las diferencias lingüísticas, gestuales y culturales de la **Lengua de Señas Salvadoreña (LESSA)**.

**neuroLESSA** surge como una solución desarrollada específicamente para reconocer y procesar señas de LESSA, utilizando visión por computadora y aprendizaje profundo entrenado con datos locales. Este proyecto busca promover la inclusión tecnológica y el respeto por la identidad lingüística de El Salvador.

### 🎯 Objetivo Principal

Desarrollar un prototipo de Inteligencia Artificial capaz de interpretar señas de LESSA en tiempo real. El sistema utiliza una cámara estándar para capturar los gestos y los convierte a **texto y voz**, facilitando la comunicación inclusiva.

### ✨ Características Clave

  * **Interpretación en Tiempo Real:** Reconocimiento instantáneo de señas.
  * **Traducción a Texto:** Muestra la palabra o frase reconocida en pantalla.
  * **Traducción a Voz:** Utiliza Google Text-to-Speech (gTTS) para verbalizar la seña.
  * **Modelo Local:** Entrenado con datos de LESSA, asegurando relevancia cultural.
  * **Interfaz Simple:** Una GUI limpia y fácil de usar construida con Tkinter.

### 🛠️ Tecnologías Utilizadas

Este proyecto se construye con las siguientes tecnologías principales:

  * **Python:** Lenguaje base del proyecto.
  * **TensorFlow / Keras:** Para el diseño y ejecución del modelo de aprendizaje profundo.
  * **OpenCV:** Para la captura y procesamiento de video en tiempo real.
  * **MediaPipe:** Para la extracción de puntos de referencia (*landmarks*) de las manos.
  * **Tkinter:** Para la construcción de la interfaz gráfica de usuario (GUI).
  * **gTTS:** Para la funcionalidad de texto a voz.
  * **scikit-learn:** Para utilidades de machine learning.

### 🚀 Instalación y Puesta en Marcha

Sigue estos pasos para poner en funcionamiento el proyecto en tu máquina local.

**1. Clona el repositorio**

```bash
git clone https://github.com/JorgeMajano/neuroLESSA.git
cd neuroLESSA
```

**2. Crea y activa un entorno virtual**

Se recomienda usar un entorno virtual para manejar las dependencias.

```bash
# Se recomienda Python 3.8, 3.9 o 3.10
python -m venv venv
```

  * En Windows:
    ```bash
    .\venv\Scripts\activate
    ```
  * En macOS/Linux:
    ```bash
    source venv/bin/activate
    ```

**3. Instala las dependencias**

El archivo `requirements.txt` contiene todas las librerías necesarias.

```bash
pip install -r requirements.txt
```

### 🏃‍♂️ Modo de Uso

Una vez que tengas el entorno configurado y las dependencias instaladas, ejecutar la aplicación es muy sencillo:

```bash
python camaraVECTORES.py
```

La aplicación se iniciará, activará tu cámara principal y abrirá la interfaz gráfica. ¡Ya puedes comenzar a hacer señas\!

### 🧑‍💻 Autores

  * **Jorge Ernesto Majano Santos**
  * **Richard Jonathan Quinteros Mendoza**

