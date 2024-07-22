import cv2
import time
import os

def tomar_fotos(cantidad_de_fotos, intervalo, directorio):
    # Crear el directorio si no existe
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    # Inicializar la captura de video
    captura = cv2.VideoCapture(0)  # 0 para la cámara predeterminada
    time.sleep(5)

    if not captura.isOpened():
        print("Error: No se puede abrir la cámara.")
        return

    for i in range(cantidad_de_fotos):
        # Tomar una foto
        ret, fotograma = captura.read()
        if not ret:
            print("Error: No se puede capturar una foto.")
            break

        # Guardar la foto en un archivo en el directorio especificado
        nombre_archivo = os.path.join(directorio, f"foto5_{i+1}.jpg")
        cv2.imwrite(nombre_archivo, fotograma)
        print(f"Foto guardada como {nombre_archivo}")

        # Esperar el intervalo antes de tomar la siguiente foto
        time.sleep(intervalo)

    # Liberar la cámara y cerrar cualquier ventana de OpenCV
    captura.release()
    cv2.destroyAllWindows()

# Número de fotos y tiempo entre fotos en segundos
cantidad_de_fotos = 900
intervalo = 0.01  # En segundos
directorio = "Entrenamiento/trans"

# Prueba si la cámara se puede abrir
def prueba_camara():
    prueba_captura = cv2.VideoCapture(0)
    if not prueba_captura.isOpened():
        print("Error: No se puede acceder a la cámara. Verifica los permisos y asegúrate de que la cámara no esté en uso.")
        return False
    ret, fotograma = prueba_captura.read()
    prueba_captura.release()
    if not ret:
        print("Error: No se puede capturar un fotograma de la cámara.")
        return False
    return True

if prueba_camara():
    tomar_fotos(cantidad_de_fotos, intervalo, directorio)
