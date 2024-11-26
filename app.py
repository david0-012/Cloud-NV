from flask import Flask, render_template, jsonify, Response
import cv2
import os
import time
import threading
from google.cloud import vision
from google.cloud.vision_v1 import types
from googletrans import Translator
from gtts import gTTS

app = Flask(__name__)

# Configuración de la cámara
cap = cv2.VideoCapture(0)

# Variable de control para detener el proceso
process_running = False

# Inicializar el cliente de traducción
translator = Translator()

# Función para traducir texto al español
def translate_to_spanish(text):
    try:
        translation = translator.translate(text, src='en', dest='es')
        return translation.text
    except Exception as e:
        print(f"Error en la traducción: {e}")
        return text  # Devuelve el texto original si ocurre un error

# Función para capturar el frame y enviarlo como un JPEG
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Codificar el frame en formato JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        # Convertir la imagen a bytes para enviarla al frontend
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Ruta para la cámara y mostrar el video en tiempo real
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Función para analizar la imagen con Google Vision API
def analyze_image(frame):
    client = vision.ImageAnnotatorClient()

    # Convertir el frame a un formato adecuado para Vision API
    _, encoded_image = cv2.imencode('.jpg', frame)
    content = encoded_image.tobytes()

    image = types.Image(content=content)
    
    # Análisis de texto (OCR)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    detected_text = texts[0].description if texts else ""

    # Análisis de objetos
    response_objects = client.object_localization(image=image)
    objects = response_objects.localized_object_annotations

    # Detectar nombres de objetos
    detected_objects = [obj.name for obj in objects]

    return detected_text, detected_objects

# Función para reproducir audio desde el texto
def play_audio(text):
    tts = gTTS(text=text, lang='es')
    tts.save("output.mp3")
    os.system("start output.mp3")  # En Windows para reproducir

# Función que ejecuta el proceso de análisis y reproducción de audio en segundo plano
def process_images():
    global process_running
    last_audio_time = time.time()  # Variable para controlar cuándo reproducir el audio

    while process_running:
        ret, frame = cap.read()
        if ret:
            detected_text, detected_objects = analyze_image(frame)

            # Traducir el texto y los objetos detectados
            translated_text = translate_to_spanish(detected_text)
            translated_objects = [translate_to_spanish(obj) for obj in detected_objects]

            # Crear el texto para el audio
            audio_text = f"{translated_text}. Objetos detectados: {', '.join(translated_objects)}"
            
            # Reproducir el audio cada 3 segundos
            if time.time() - last_audio_time >= 3:
                play_audio(audio_text)
                last_audio_time = time.time()  # Actualizar el tiempo de la última reproducción
        
        time.sleep(1)  # Agregar un pequeño retraso para no sobrecargar la CPU

# Ruta principal para la interfaz web
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para ejecutar el procesamiento de cámara (iniciar el proceso)
@app.route('/start_process', methods=['GET'])
def start_process():
    global process_running

    if not process_running:
        process_running = True
        # Ejecutar el proceso de imágenes en un hilo en segundo plano
        threading.Thread(target=process_images, daemon=True).start()
        return jsonify({"message": "Proceso iniciado, espere para escuchar los resultados."})
    else:
        return jsonify({"message": "El proceso ya está en ejecución."})

# Ruta para detener el proceso
@app.route('/stop_process', methods=['GET'])
def stop_process():
    global process_running
    process_running = False
    return jsonify({"message": "Proceso detenido."})

if __name__ == '__main__':
    app.run(debug=True)
