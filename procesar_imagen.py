import sys
import face_recognition
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

if len(sys.argv) < 3:
    print("Uso: procesar_imagen.py <ruta_imagen> <nombre>")
    sys.exit(1)

image_path = sys.argv[1]
nombre = sys.argv[2]

# Cargar imagen
image = face_recognition.load_image_file(image_path)

# Extraer vectores faciales
face_encodings = face_recognition.face_encodings(image)

print(f"Se detectaron {len(face_encodings)} rostros en la imagen.")

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cursor = conn.cursor()

if len(face_encodings) > 0:
    for encoding in face_encodings:
        cursor.execute(
            "INSERT INTO rostros (nombre, vector) VALUES (%s, %s)",
            (nombre, [float(x) for x in encoding])
        )
    conn.commit()
    print(f"Vectores faciales guardados para {nombre}.")
else:
    print("No se detectaron rostros.")

cursor.close()
conn.close()
