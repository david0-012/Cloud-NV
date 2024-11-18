from flask import Flask, request, render_template, send_from_directory
import os
import boto3
from google.cloud import translate_v2 as translate
import azure.cognitiveservices.speech as speechsdk

app = Flask(__name__)

# Asegúrate de que la carpeta 'static' exista
if not os.path.exists('static'):
    os.makedirs('static')

# Configura los clientes de Rekognition, S3, Textract y Translate
rekognition_client = boto3.client('rekognition', region_name='us-east-1')
s3_client = boto3.client('s3', region_name='us-east-1')
translate_client = translate.Client()  # Asegúrate de tener la variable de entorno configurada
textract_client = boto3.client('textract', region_name='us-east-1')  # Cliente para Textract

# Configuración de Azure Cognitive Services (Speech API)
AZURE_SPEECH_KEY = "F1cNfL9HAr6H2BRY5RHvdMPdBOe3ZPSlhIQLxzrr7g45mFn2ZKiRJQQJ99AKACYeBjFXJ3w3AAAYACOGzdLG"
AZURE_REGION = "eastus"  # Cambia esto si tu región es diferente

# Función para traducir texto
def translate_text(text, target='es'):
    if not text:
        return ""
    result = translate_client.translate(text, target_language=target)
    return result['translatedText']

# Función para convertir texto a voz y guardarlo en un archivo
def text_to_speech(text, filename):
    if not text:
        return "No text to convert to speech"

    # Crear un objeto de configuración de voz
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)

    # Configura la salida de audio a un archivo
    audio_output_path = os.path.join('static', filename)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_output_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # Sintetizar el texto a voz y guardar en el archivo
    result = synthesizer.speak_text_async(text).get()  # Asegúrate de que el proceso termine antes de continuar

    # Verificar si hubo algún error al sintetizar
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        return f"Error al sintetizar el audio: {result.error_details}"

    return audio_output_path  # Devuelve la ruta del archivo de audio

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para manejar la carga de la imagen y el procesamiento
@app.route('/upload', methods=['POST'])
def upload():
    # Obtener el idioma seleccionado del formulario
    target_language = request.form.get('language', 'es')  # Por defecto es español
    
    if 'image' not in request.files:
        return "No file part", 400
    
    file = request.files['image']
    
    if file.filename == '':
        return "No selected file", 400
    
    # Guarda la imagen temporalmente
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    # Cambia esto al nombre de tu bucket
    bucket_name = 'mi-bucket-imagenes-juan'
    image_name = file.filename

    # Sube la imagen a S3
    try:
        s3_client.upload_file(file_path, bucket_name, image_name)
    except Exception as e:
        return f"Error uploading file: {str(e)}", 500

    # Identificar objetos en la imagen usando Rekognition
    try:
        rekognition_response = rekognition_client.detect_labels(
            Image={'S3Object': {'Bucket': bucket_name, 'Name': image_name}} ,
            MaxLabels=10,
            MinConfidence=75
        )
        
        labels = [label['Name'] for label in rekognition_response['Labels']]
    except Exception as e:
        return f"Error in Rekognition: {str(e)}", 500

    # Extraer texto de la imagen usando Textract
    try:
        with open(file_path, 'rb') as document:
            response = textract_client.detect_document_text(
                Document={'Bytes': document.read()}
            )
        
        extracted_text_lines = []
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                extracted_text_lines.append(item['Text'])
        
        extracted_text = "\n".join(extracted_text_lines)
    except Exception as e:
        return f"Error in Textract: {str(e)}", 500

    # Traducir las etiquetas de objetos al idioma seleccionado
    translated_labels = []
    for label in labels:
        try:
            translated_label = translate_text(label, target=target_language)
            translated_labels.append({
                "original": label,
                "translated": translated_label
            })
        except Exception as e:
            return f"Error in translation for labels: {str(e)}", 500

    # Traducir el texto extraído como un todo
    try:
        translated_extracted_text = translate_text(extracted_text, target=target_language)
    except Exception as e:
        return f"Error in translation for extracted text: {str(e)}", 500

    # Guardar el audio de la traducción en un archivo, con un nombre único
    audio_filename = f"translated_text_audio_{file.filename}.wav"
    audio_path = text_to_speech(translated_extracted_text, audio_filename)

    return render_template('index.html', 
                           identified_objects=translated_labels, 
                           extracted_text=extracted_text, 
                           translated_extracted_text=translated_extracted_text,
                           audio_file=audio_filename)

# Ruta para servir el archivo de audio
@app.route('/audio/<filename>')
def audio(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
