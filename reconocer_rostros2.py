import face_recognition
import cv2
import psycopg2
from dotenv import load_dotenv
import os
import numpy as np

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

# Obtener todos los vectores y nombres de la tabla vectores_faciales uniendo con usuarios
cursor.execute("""
    SELECT u.nombre_completo, v.vector
    FROM vectores_faciales v
    JOIN usuarios u ON v.id_usuario = u.id_usuario
""")
registros = cursor.fetchall()

nombres_conocidos = []
vectores_conocidos = []

# Agrupar vectores por nombre para soportar múltiples vectores por usuario
for nombre, vector in registros:
    nombres_conocidos.append(nombre)
    vectores_conocidos.append(np.array(vector, dtype=np.float64))

print(f"🧠 Se cargaron {len(vectores_conocidos)} vectores faciales conocidos.")

# Iniciar cámara
video_capture = cv2.VideoCapture(0)

print("🔍 Reconociendo rostros en tiempo real... Presiona 'Q' para salir.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Reducir tamaño para acelerar procesamiento (opcional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detectar rostros y vectores faciales
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        # Comparar con todos los vectores conocidos con una tolerancia ajustable
        matches = face_recognition.compare_faces(vectores_conocidos, encoding, tolerance=0.5)
        nombre = "Desconocido"

        # Calcular distancias para el mejor match
        face_distances = face_recognition.face_distance(vectores_conocidos, encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            nombre = nombres_conocidos[best_match_index]

        # Ajustar coordenadas al tamaño original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Dibujar cuadro y nombre
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow('Reconocimiento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
