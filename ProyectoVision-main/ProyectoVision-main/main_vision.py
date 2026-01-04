import os
import sys
import subprocess
import time
from funciones.version_final_exporta_fotos import analyze_camera_snapshot, analyze_image

# Configuración
IMAGE_PATH = os.path.join("imagenes", "imagen_test.jpg")
# POINTS_JSON_PATH se calculará dinámicamente
ARUCO_MM = 50
ARUCO_SIZE = "AUTO"

def run_pipeline(use_camera=True):
    print("\n" + "="*50)
    print("🚀 INICIANDO PIPELINE DE VISIÓN")
    print("="*50)

    # Determinar rutas dinámicamente
    if use_camera:
        current_image = os.path.join("imagenes", "captura_camara.png")
        current_json = os.path.join("puntos", "captura_camara_puntos.json")
    else:
        current_image = IMAGE_PATH
        if not os.path.exists(current_image):
             print(f"❌ Error: No existe la imagen {current_image}")
             return
        
        base_name = os.path.splitext(os.path.basename(current_image))[0]
        current_json = os.path.join("puntos", f"{base_name}_puntos.json")

    print(f"ℹ️  Procesando imagen: {current_image}")
    print(f"ℹ️  Archivo de puntos esperado: {current_json}")

    # 1. CAPTURA / ANÁLISIS INICIAL
    print("\n[PASO 1] Captura y Detección de Objetos (version_final_exporta_fotos.py)")
    try:
        if use_camera:
            # Usa la función importada que ya maneja la cámara y guarda en imagenes/captura_camara.png
            analyze_camera_snapshot()
        else:
            # Usa la imagen existente
            analyze_image(current_image)
    except Exception as e:
        print(f"❌ Error en Paso 1: {e}")
        return

    # 2. DETECCIÓN DE CAJAS NEGRAS
    print("\n[PASO 2] Detección de Contenedores (detector_rectangulos_negros.py)")
    detector_script = os.path.join("funciones", "detector_rectangulos_negros.py")
    detector_save = os.path.join("imagenes", "resultado_deteccion_cajas.png")
    
    cmd_detector = [
        sys.executable, detector_script,
        "--image", current_image,
        "--save", detector_save
    ]
    
    try:
        subprocess.run(cmd_detector, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en Paso 2: {e}")
        return

    # 3. MEDICIÓN Y CONVERSIÓN A MM
    print("\n[PASO 3] Medición y Conversión a mm (aruco_batch_measure.py)")
    measure_script = os.path.join("funciones", "aruco_batch_measure.py")
    
    cmd_measure = [
        sys.executable, measure_script,
        "--image", current_image,
        "--points-json", current_json,
        "--aruco-mm", str(ARUCO_MM),
        "--aruco-tamano", ARUCO_SIZE
    ]
    
    try:
        subprocess.run(cmd_measure, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en Paso 3: {e}")
        return

    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"📂 Resultados en carpeta 'puntos/' y 'imagenes/'")
    print("="*50)

def main():
    while True:
        print("\n--- MENÚ PRINCIPAL ---")
        print("1. Ejecutar Pipeline con CÁMARA")
        print("2. Ejecutar Pipeline con IMAGEN EXISTENTE (captura_camara.png)")
        print("3. Salir")
        
        opcion = input("Selecciona una opción: ").strip()
        
        if opcion == "1":
            run_pipeline(use_camera=True)
        elif opcion == "2":
            run_pipeline(use_camera=False)
        elif opcion == "3":
            print("👋 Saliendo...")
            break
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    main()
