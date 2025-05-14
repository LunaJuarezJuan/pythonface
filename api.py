from flask import Flask, request, jsonify
import face_recognition
import cv2
import psycopg2
import numpy as np
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Conectar a la base de datos
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cursor = conn.cursor()

# Inicializar Flask
app = Flask(__name__)

@app.route('/guardar_vectores', methods=['POST'])
def guardar_vectores():
    # Se espera recibir una imagen codificada en base64
    data = request.json
    image_data = data['image']  # Asume que la imagen está en base64
    nombre = data['nombre']

    # Decodificar la imagen base64
    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if face_encodings:
        for encoding in face_encodings:
            cursor.execute("INSERT INTO rostros (nombre, vector) VALUES (%s, %s)", (nombre, [float(x) for x in encoding]))
        conn.commit()
        return jsonify({"message": f"✅ {nombre} guardado con {len(face_encodings)} vector(es).", "status": "success"}), 200
    else:
        return jsonify({"message": "⚠️ No se detectó ningún rostro.", "status": "error"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
