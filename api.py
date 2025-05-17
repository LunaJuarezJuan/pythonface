import sys
import face_recognition
import psycopg2
from dotenv import load_dotenv
import os
import hashlib

load_dotenv()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_or_create_usuario(nombre, correo, contrasena, cursor, conn):
    cursor.execute("SELECT id_usuario FROM usuarios WHERE correo = %s", (correo,))
    result = cursor.fetchone()
    if result:
        return result[0]

    contrasena_hash = hash_password(contrasena)
    cursor.execute(
        "INSERT INTO usuarios (nombre_completo, correo, contrasena_hash) VALUES (%s, %s, %s) RETURNING id_usuario",
        (nombre, correo, contrasena_hash)
    )
    id_usuario = cursor.fetchone()[0]
    conn.commit()
    return id_usuario

def guardar_vectores(id_usuario, vectores, cursor, conn):
    for encoding in vectores:
        cursor.execute(
            "INSERT INTO vectores_faciales (id_usuario, vector) VALUES (%s, %s)",
            (id_usuario, [float(x) for x in encoding])
        )
    conn.commit()

def main():
    if len(sys.argv) < 5:
        print("Uso: python api.py <ruta_imagen> <nombre_completo> <correo> <contrasena>")
        sys.exit(1)

    image_path = sys.argv[1]
    nombre = sys.argv[2]
    correo = sys.argv[3]
    contrasena = sys.argv[4]

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    cursor = conn.cursor()

    id_usuario = get_or_create_usuario(nombre, correo, contrasena, cursor, conn)
    print(f"Usuario con id {id_usuario} listo.")

    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    print(f"Se detectaron {len(face_encodings)} rostros.")

    if face_encodings:
        guardar_vectores(id_usuario, face_encodings, cursor, conn)
        print(f"Vectores guardados para usuario {nombre}.")
    else:
        print("No se detectaron rostros.")

    cursor.close()
    conn.close()
    
if __name__ == "__main__":
    main()
