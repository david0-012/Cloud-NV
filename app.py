from flask import Flask, render_template, jsonify, Response
import cv2
import os
import time
import threading
from google.cloud import vision
from google.cloud.vision_v1 import types
from gtts import gTTS

app = Flask(__name__)

# Configuración de la cámara
cap = cv2.VideoCapture(0)

# Diccionario de traducción de objetos al español
object_translation = {
    "Person": "Persona",
    "Cat": "Gato",
    "Dog": "Perro",
    "Car": "Coche",
    "Bicycle": "Bicicleta",
    "Jacket": "Chaqueta",
    "Glasses": "Gafas"
}

# Función para traducir texto al español (puedes añadir un servicio de traducción real si lo deseas)
def translate_to_spanish(text):
    return text  # Aquí puedes agregar la traducción si es necesario.

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

    # Análisis de objetos
    response_objects = client.object_localization(image=image)
    objects = response_objects.localized_object_annotations

    text = texts[0].description if texts else ""
    
    # Traducir los objetos al español usando el diccionario
    detected_objects = [object_translation.get(obj.name, obj.name) for obj in objects]

    # Traducir el texto detectado al español
    translated_text = translate_to_spanish(text)

    return translated_text, detected_objects

# Función para reproducir audio desde el texto
def play_audio(text):
    tts = gTTS(text=text, lang='es')
    tts.save("output.mp3")
    os.system("start output.mp3")  # En Windows para reproducir

# Ruta principal para la interfaz web
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para ejecutar el procesamiento de cámara
@app.route('/start_process', methods=['GET'])
def start_process():
    start_time = time.time()
    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if ret:
            text, detected_objects = analyze_image(frame)
            # Concatenar todo el texto en español, incluyendo el texto detectado y los objetos
            # Asegurarnos de que todos los componentes estén en español
            audio_text = f"{translate_to_spanish(text)}. Objetos detectados: {', '.join(detected_objects)}"
            play_audio(audio_text)
        time.sleep(1)
    
    return jsonify({"message": "Proceso iniciado, espere 3 segundos para escuchar los resultados."})

if __name__ == '__main__':
    app.run(debug=True)
