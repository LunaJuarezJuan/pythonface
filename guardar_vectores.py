# guardar_vectores.py
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

# Captura desde cámara
video_capture = cv2.VideoCapture(0)

print("📷 Presiona 'S' para guardar rostro, 'Q' para salir.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow('Guardar Rostro - Presiona S', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') or key == ord('S'):
        if face_encodings:
            nombre = input("🧑 Ingresa el nombre de la persona: ")
            for encoding in face_encodings:
                cursor.execute("INSERT INTO rostros (nombre, vector) VALUES (%s, %s)", (nombre, [float(x) for x in encoding]))

            conn.commit()
            print(f"✅ {nombre} guardado con {len(face_encodings)} vector(es).")
        else:
            print("⚠️ No se detectó ningún rostro.")

    elif key == ord('q') or key == ord('Q'):
        break

video_capture.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
