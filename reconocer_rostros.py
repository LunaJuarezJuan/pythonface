# reconocer_rostros.py
import face_recognition
import cv2
import psycopg2
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

# Obtener todos los vectores y nombres
cursor.execute("SELECT nombre, vector FROM rostros")
registros = cursor.fetchall()

nombres_conocidos = []
vectores_conocidos = []

for nombre, vector in registros:
    nombres_conocidos.append(nombre)
    vectores_conocidos.append(vector)

print(f"🧠 Se cargaron {len(vectores_conocidos)} rostros conocidos.")

# Iniciar cámara
video_capture = cv2.VideoCapture(0)

print("🔍 Reconociendo rostros... Presiona Q para salir.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(vectores_conocidos, encoding, tolerance=0.5)
        nombre = "Desconocido"

        if True in matches:
            match_index = matches.index(True)
            nombre = nombres_conocidos[match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow('Reconocimiento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
