import os
import boto3
from flask import Flask, request, render_template
from google.cloud import translate_v2 as translate

app = Flask(__name__)

# Asegúrate de que la carpeta 'static' exista
if not os.path.exists('static'):
    os.makedirs('static')

# Configura los clientes de Rekognition, S3, Textract y Translate
rekognition_client = boto3.client('rekognition', region_name='us-east-1')
s3_client = boto3.client('s3', region_name='us-east-1')
translate_client = translate.Client()  # Asegúrate de tener la variable de entorno configurada
textract_client = boto3.client('textract', region_name='us-east-1')  # Cliente para Textract

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Función para traducir texto
def translate_text(text, target='es'):
    if not text:
        return ""
    result = translate_client.translate(text, target_language=target)
    return result['translatedText']

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
            Image={'S3Object': {'Bucket': bucket_name, 'Name': image_name}},
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
        
        # Agrupar líneas en párrafos (puedes modificar esto según tus necesidades)
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

    return render_template('index.html', identified_objects=translated_labels, extracted_text=extracted_text, translated_extracted_text=translated_extracted_text)

if __name__ == '__main__':
    app.run(debug=True)
