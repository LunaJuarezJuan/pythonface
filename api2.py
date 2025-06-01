import face_recognition
import cv2
import numpy as np
import psycopg2
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# Conexión a la DB
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

cursor = conn.cursor()

def obtener_sesion_activa():
    ahora = datetime.now()
    cursor.execute("""
        SELECT id_sesion, id_clase, hora_inicio, hora_fin
        FROM sesiones
        WHERE hora_inicio <= %s AND hora_fin >= %s
        LIMIT 1
    """, (ahora, ahora))
    return cursor.fetchone()  # (id_sesion, id_clase, hora_inicio, hora_fin) o None

def obtener_alumnos_inscritos(id_sesion):
    cursor.execute("""
        SELECT u.id_usuario, u.nombre_completo
        FROM inscripciones i
        JOIN usuarios u ON i.id_usuario = u.id_usuario
        WHERE i.id_sesion = %s
    """, (id_sesion,))
    return cursor.fetchall()  # Lista de (id_usuario, nombre_completo)

def cargar_vectores_alumnos(id_usuarios):
    vectores = {}
    for uid in id_usuarios:
        cursor.execute("""
            SELECT vector FROM vectores_faciales WHERE id_usuario = %s
        """, (uid,))
        rows = cursor.fetchall()
        # Convertir vectores guardados a listas numpy
        vectores[uid] = [np.array(row[0]) for row in rows]
    return vectores

def guardar_asistencia(id_sesion, id_usuario, estado):
    ahora = datetime.now()
    cursor.execute("""
        INSERT INTO asistencia_sesion (id_sesion, id_usuario, estado, fecha_registro)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id_sesion, id_usuario, fecha_registro) DO UPDATE SET estado = EXCLUDED.estado
    """, (id_sesion, id_usuario, estado, ahora))
    conn.commit()

def reconocer_rostros(vectores_alumnos):
    video_capture = cv2.VideoCapture(0)  # Cámara en vivo, o cambiar por archivo

    alumnos_presentes = set()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Redimensionar para acelerar procesamiento
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            for uid, lista_vectores in vectores_alumnos.items():
                matches = face_recognition.compare_faces(lista_vectores, face_encoding)
                if True in matches:
                    alumnos_presentes.add(uid)
                    break

        # Mostrar la ventana con detección (opcional)
        cv2.imshow('Reconocimiento Facial', frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return alumnos_presentes

def main():
    sesion = obtener_sesion_activa()
    if not sesion:
        print("No hay sesión activa en este momento.")
        return

    id_sesion, id_clase, hora_inicio, hora_fin = sesion
    print(f"Sesión activa: {id_sesion} - Clase: {id_clase}")

    alumnos = obtener_alumnos_inscritos(id_sesion)
    if not alumnos:
        print("No hay alumnos inscritos en esta sesión.")
        return

    id_usuarios = [a[0] for a in alumnos]
    vectores_alumnos = cargar_vectores_alumnos(id_usuarios)

    print(f"Detectando asistencia para {len(alumnos)} alumnos...")

    alumnos_presentes = reconocer_rostros(vectores_alumnos)

    ahora = datetime.now()
    limite_tarde = hora_inicio + timedelta(minutes=10)  # 10 minutos de tolerancia

    for uid in id_usuarios:
        if uid in alumnos_presentes:
            estado = "presente" if ahora <= limite_tarde else "tarde"
        else:
            estado = "ausente"

        guardar_asistencia(id_sesion, uid, estado)
        print(f"Alumno {uid}: {estado}")

if __name__ == "__main__":
    main()
