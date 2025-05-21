#controlar la asistencia

import face_recognition
import cv2
import psycopg2
from dotenv import load_dotenv
import os
import time
import numpy as np
from datetime import datetime

load_dotenv()

class AlumnoEstado:
    def __init__(self):
        self.hora_ingreso = None
        self.tiempo_presente = 0
        self.tiempo_ausente = 0
        self.esta_presente = False
        self.asistencia = False
        self.tardanza = False

class AsistenciaMonitor:
    def __init__(self, hora_inicio_clase, max_tardanza=600, max_ausencia=480):
        self.hora_inicio_clase = hora_inicio_clase  # epoch en segundos
        self.max_tardanza = max_tardanza            # segundos (10 min)
        self.max_ausencia = max_ausencia            # segundos (8 min)
        self.alumnos = {}  # dict nombre -> AlumnoEstado

    def actualizar(self, nombres_detectados):
        ahora = time.time()

        # Actualizar estado para alumnos detectados
        for nombre in nombres_detectados:
            estado = self.alumnos.get(nombre, AlumnoEstado())
            if estado.hora_ingreso is None:
                estado.hora_ingreso = ahora
                if ahora - self.hora_inicio_clase > self.max_tardanza:
                    estado.tardanza = True
                else:
                    estado.tardanza = False
            estado.esta_presente = True
            estado.tiempo_ausente = 0
            estado.tiempo_presente += 1
            if not estado.tardanza:
                estado.asistencia = True
            self.alumnos[nombre] = estado

        # Marcar ausentes
        alumnos_previos = set(self.alumnos.keys())
        ausentes = alumnos_previos - set(nombres_detectados)

        for nombre in ausentes:
            estado = self.alumnos[nombre]
            if estado.esta_presente:
                estado.tiempo_ausente += 1
                if estado.tiempo_ausente > self.max_ausencia:
                    estado.asistencia = False
                    estado.esta_presente = False
            self.alumnos[nombre] = estado

    def obtener_reporte(self):
        reporte = {}
        for nombre, estado in self.alumnos.items():
            reporte[nombre] = {
                'presente': estado.esta_presente,
                'asistencia': estado.asistencia,
                'tardanza': estado.tardanza,
                'tiempo_presente_segundos': estado.tiempo_presente,
                'tiempo_ausente_segundos': estado.tiempo_ausente
            }
        return reporte

def guardar_asistencia_en_db(conn, monitor):
    fecha_sesion = datetime.now()
    cursor = conn.cursor()

    reporte = monitor.obtener_reporte()
    for nombre, estado in reporte.items():
        cursor.execute("SELECT id_usuario FROM usuarios WHERE nombre_completo = %s", (nombre,))
        result = cursor.fetchone()
        if result:
            id_usuario = result[0]
            cursor.execute("""
                INSERT INTO asistencia_sesion (id_usuario, fecha_sesion, presente, tardanza, tiempo_presente, tiempo_ausente)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (id_usuario, fecha_sesion, estado['asistencia'], estado['tardanza'], estado['tiempo_presente_segundos'], estado['tiempo_ausente_segundos']))
    conn.commit()
    cursor.close()

def main():
    # Conexión base de datos
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.nombre_completo, v.vector
        FROM vectores_faciales v
        JOIN usuarios u ON v.id_usuario = u.id_usuario
    """)
    registros = cursor.fetchall()

    nombres_conocidos = []
    vectores_conocidos = []

    for nombre, vector in registros:
        nombres_conocidos.append(nombre)
        vectores_conocidos.append(np.array(vector, dtype=np.float64))

    print(f"🧠 Se cargaron {len(vectores_conocidos)} vectores faciales conocidos.")

    video_capture = cv2.VideoCapture(0)

    monitor = AsistenciaMonitor(hora_inicio_clase=time.time())

    tiempo_sesion_segundos = 60 * 60  # 60 minutos, cambia según duración clase
    inicio = time.time()

    print("🔍 Iniciando monitoreo de asistencia. Presiona 'q' para finalizar antes.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        nombres_detectados = set()

        for encoding in face_encodings:
            matches = face_recognition.compare_faces(vectores_conocidos, encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(vectores_conocidos, encoding)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

            nombre = "Desconocido"
            if best_match_index is not None and matches[best_match_index]:
                nombre = nombres_conocidos[best_match_index]

            if nombre != "Desconocido":
                nombres_detectados.add(nombre)

        monitor.actualizar(nombres_detectados)

        # Mostrar en consola (puedes eliminar o ajustar para menos info)
        reporte = monitor.obtener_reporte()
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Reporte parcial de asistencia:")
        for nombre, estado in reporte.items():
            print(f"{nombre}: Presente={estado['presente']}, Asistencia={estado['asistencia']}, "
                  f"Tardanza={estado['tardanza']}, Tiempo Presente={estado['tiempo_presente_segundos']}s, "
                  f"Tiempo Ausente={estado['tiempo_ausente_segundos']}s")

        # Mostrar ventana cámara con cuadros y nombres
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(vectores_conocidos, encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(vectores_conocidos, encoding)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

            nombre = "Desconocido"
            if best_match_index is not None and matches[best_match_index]:
                nombre = nombres_conocidos[best_match_index]

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.imshow('Monitoreo de Asistencia', frame)

        # Finalizar por tiempo o tecla
        if (time.time() - inicio) > tiempo_sesion_segundos or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    # Guardar asistencia en BD
    guardar_asistencia_en_db(conn, monitor)

    video_capture.release()
    cv2.destroyAllWindows()
    cursor.close()
    conn.close()
    print("Monitoreo finalizado y asistencia guardada.")

if __name__ == "__main__":
    main()
