import cv2
import numpy as np
import time
import os
import json

# =========================
# Parámetros ajustables
# =========================
CAMERA_ID = 1  # cambia aquí tu índice si es necesario

# --- VARIABLE DE CONTROL DE LUZ (La que tú mandas) ---
# El filtro oscurecerá la imagen hasta llegar a este valor exacto.
TARGET_REAL_BRIGHTNESS = 60
# --- Ajuste Automático (Interruptor general)
ENABLE_AUTO_LIGHT = True

# --- Tapete (oscuro en gris)
MAT_G_MAX = 150         
MAT_INNER_MARGIN = 0    

# --- Objetos (claro sobre negro)
DELTA_G = 20          
MIN_OBJ_AREA = 1800
SOLIDITY_MIN = 0.68
EXTENT_MIN = 0.30     
CIRC_MIN = 0.28       
MIN_SIDE = 18         
MAX_EXCENTRICITY = 6.0  

# --- CLASIFICACIÓN DE FORMA ---
RECT_THRESHOLD = 0.86 

# --- Detección de Tapones (Círculos)
CAP_MIN_RADIUS = 15   
CAP_MAX_RADIUS = 50   
HOUGH_PARAM1 = 50     
HOUGH_PARAM2 = 25     

DRAW_GRIP = True  

# Carpeta para guardar puntos
PUNTOS_DIR = "puntos"

# =========================
# Apertura de cámara robusta
# =========================
def open_cam(cam_id):
    for backend in (cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cap = cv2.VideoCapture(cam_id, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(cam_id)

# =========================
# LÓGICA DE FILTRO (CONTRASTE + TARGET REAL)
# =========================
def apply_target_brightness_filter(img):
    """
    Aplica contraste y reduce brillo hasta llegar a TARGET_REAL_BRIGHTNESS.
    """
    if not ENABLE_AUTO_LIGHT:
        return img
    
    print(f"--- Iniciando Ajuste hacia Objetivo: {TARGET_REAL_BRIGHTNESS} ---")

    # 1. APLICAR CONTRASTE INICIAL
    # alpha 1.3 = 30% más contraste
    img = cv2.convertScaleAbs(img, alpha=1.3, beta=0)

    # 2. BUCLE DE AJUSTE
    max_iterations = 100 
    
    for i in range(max_iterations):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi_temp = get_mat_mask(gray)
        
        # Seguridad: Si el tapete desaparece, paramos
        if roi_temp.sum() == 0: 
            print("! Tapete perdido (demasiado oscuro).")
            break

        current_brightness = compute_brightness(gray, roi_temp)
        
        # === AQUÍ USAMOS TU VARIABLE ===
        if current_brightness <= (TARGET_REAL_BRIGHTNESS + 0.5):
            print(f"✅ Brillo alcanzado: {current_brightness:.2f} (Objetivo: {TARGET_REAL_BRIGHTNESS})")
            break
        
        diff = current_brightness - TARGET_REAL_BRIGHTNESS
        
        # Acelerador: si estamos muy lejos, bajamos más rápido
        resta = int(diff) if diff > 5 else 1
        
        # Restar luz
        img = img.astype(np.int16)
        img = img - resta
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img

# =========================
# Buscador de Tapones
# =========================
def find_cap_center(gray_roi, offset_x, offset_y):
    blurred = cv2.GaussianBlur(gray_roi, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=20,
        param1=HOUGH_PARAM1, 
        param2=HOUGH_PARAM2, 
        minRadius=CAP_MIN_RADIUS, 
        maxRadius=CAP_MAX_RADIUS
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        best_circle = circles[0] 
        cx, cy, r = best_circle
        global_cx = cx + offset_x
        global_cy = cy + offset_y
        return (global_cx, global_cy), r
    return None, None

# =========================
# Funciones de Visión
# =========================
def get_mat_mask(gray):
    _, m = cv2.threshold(gray, MAT_G_MAX, 255, cv2.THRESH_BINARY_INV)
    m = cv2.medianBlur(m, 5)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), 2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return np.zeros_like(m)
    biggest = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(m)
    cv2.drawContours(mask, [biggest], -1, 255, thickness=cv2.FILLED)
    if MAT_INNER_MARGIN > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*MAT_INNER_MARGIN+1, 2*MAT_INNER_MARGIN+1))
        mask = cv2.erode(mask, k, iterations=1)
    return mask

def objects_mask(gray, roi_mask):
    roi_vals = gray[roi_mask > 0]
    if roi_vals.size == 0: return np.zeros_like(roi_mask)
    g_med = np.median(roi_vals)
    rel = cv2.threshold(gray, int(g_med + DELTA_G), 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    th = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask = cv2.bitwise_or(rel, th)
    mask = cv2.bitwise_and(mask, roi_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    return mask

def good_shape(cnt):
    area = cv2.contourArea(cnt)
    if area < MIN_OBJ_AREA: return False
    x, y, w, h = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (W, H), ang = rect
    smin, smax = min(W, H), max(W, H)
    if smin < MIN_SIDE: return False
    if (smax/(smin+1e-9)) > MAX_EXCENTRICITY: return False
    return True

def raycast(mask, center, direction, step=1.0, max_steps=4000):
    x, y = float(center[0]), float(center[1])
    H, W = mask.shape[:2]
    for _ in range(max_steps):
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= W or yi >= H: break
        if mask[yi, xi] == 0:
            x -= direction[0] * step
            y -= direction[1] * step
            return int(round(x)), int(round(y))
        x += direction[0] * step
        y += direction[1] * step
    return int(round(x)), int(round(y))

def grip_from_minarea(cnt, mask, override_center=None):
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (W, H), ang = rect
    if override_center is not None:
        cx, cy = override_center
    if W < H:
        ang += 90.0
        W, H = H, W
    th = np.deg2rad(ang)
    dx, dy = np.cos(th + np.pi/2), np.sin(th + np.pi/2)
    p1 = raycast(mask, (cx, cy), (dx, dy))
    p2 = raycast(mask, (cx, cy), (-dx, -dy))
    return (int(cx), int(cy)), p1, p2

def compute_brightness(gray, roi_mask):
    vals = gray[roi_mask > 0]
    return float(np.mean(vals)) if vals.size > 0 else 0.0

def draw_light_bar(img, current, target):
    h, w = img.shape[:2]
    bar_w, bar_h, margin = 180, 18, 15
    x2, x1 = w - margin, w - margin - bar_w
    y1, y2 = margin, margin + bar_h
    cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 22), (20, 20, 20), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)
    cur_ratio = np.clip(current / 255.0, 0.0, 1.0)
    cv2.rectangle(img, (x1, y1), (int(x1 + cur_ratio * bar_w), y2), (180, 180, 180), -1)
    tgt_ratio = np.clip(target / 255.0, 0.0, 1.0)
    tx = int(x1 + tgt_ratio * bar_w)
    cv2.line(img, (tx, y1 - 2), (tx, y2 + 2), (230, 230, 230), 2)
    cv2.putText(img, f"Luz: {int(current):3d} / {int(target):3d}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

def enumerate_contours(contours, img_h):
    cell = max(40, img_h // 15)
    def key(c):
        M = cv2.moments(c)
        cx = M['m10']/(M['m00'] + 1e-9)
        cy = M['m01']/(M['m00'] + 1e-9)
        return (int(cy // cell), cx)
    return sorted(contours, key=key)

# =========================
# LÓGICA DE DETECCIÓN PRINCIPAL
# =========================
def get_detections(gray_original):
    detections = []
    roi = get_mat_mask(gray_original)
    if roi is None or roi.sum() == 0:
        return detections, np.zeros(gray_original.shape[:2], dtype=np.uint8)
    
    mask = objects_mask(gray_original, roi)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if good_shape(c)]
    cnts = enumerate_contours(cnts, gray_original.shape[0])
    
    for i, c in enumerate(cnts, start=1):
        m = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 255, thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        cx, cy = int(M['m10']/(M['m00'] + 1e-9)), int(M['m01']/(M['m00'] + 1e-9))
        
        # --- CLASIFICACIÓN (Rect vs Cyl) ---
        rect_min = cv2.minAreaRect(c)
        w_r, h_r = rect_min[1]
        area_min_rect = w_r * h_r
        area_contour = cv2.contourArea(c)
        
        rectangularity = area_contour / (area_min_rect + 1e-9)
        is_rectangular = rectangularity > RECT_THRESHOLD
        
        obj_type = "RECT" if is_rectangular else "CYL"
        
        cap_center = None
        cap_radius = None
        
        if not is_rectangular:
            obj_roi = gray_original[y:y+h, x:x+w]
            obj_mask_roi = m[y:y+h, x:x+w]
            masked_obj = cv2.bitwise_and(obj_roi, obj_roi, mask=obj_mask_roi)
            
            cap_center, cap_radius = find_cap_center(masked_obj, x, y)
        
        override = cap_center if (not is_rectangular and cap_center is not None) else None
        
        center, p1, p2 = (None, None, None)
        if DRAW_GRIP:
            center, p1, p2 = grip_from_minarea(c, m, override_center=override)
            
        detections.append({
            "id": i, "bbox": (x, y, w, h), "centroid": (cx, cy), "contour": c,
            "mask": m, "grip_center": center, "grip_p1": p1, "grip_p2": p2,
            "type": obj_type,
            "rectangularity": rectangularity,
            "has_cap": (cap_center is not None), "cap_radius": cap_radius
        })
    return detections, roi

def process_frame(frame_bgr):
    # AQUI LLAMAMOS A LA NUEVA FUNCION
    frame_corrected = apply_target_brightness_filter(frame_bgr)
    
    gray = cv2.cvtColor(frame_corrected, cv2.COLOR_BGR2GRAY)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    detections, roi = get_detections(gray)
    
    if roi is None or (isinstance(roi, np.ndarray) and roi.sum() == 0):
        cv2.putText(out, "No se detecta tapete", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
        return out
        
    ov = out.copy()
    ov[roi > 0] = (0, 0, 0)
    out = cv2.addWeighted(ov, 0.25, out, 0.75, 0)
    
    for det in detections:
        c = det["contour"]
        x, y, w, h = det["bbox"]
        
        color_txt = (255, 255, 0) if det["type"] == "CYL" else (255, 200, 100)
        color_contorno = (0, 255, 0) if det["has_cap"] else (255, 255, 255)
        
        cv2.drawContours(out, [c], -1, color_contorno, 2)
        
        if DRAW_GRIP and det["grip_center"] is not None:
            cv2.line(out, det["grip_p1"], det["grip_p2"], (220, 220, 220), 3)
            cv2.circle(out, det["grip_p1"], 5, (0, 0, 0), -1)
            cv2.circle(out, det["grip_p2"], 5, (0, 0, 0), -1)
            cv2.circle(out, det["grip_center"], 6, (0, 0, 255), -1) 
            
            if det["has_cap"]:
                 radius = det["cap_radius"] if det["cap_radius"] else 15
                 cv2.circle(out, det["grip_center"], int(radius), (0, 255, 255), 1)

        cv2.rectangle(out, (x, y - 40), (x + 100, y - 6), (0, 0, 0), -1)
        label = f"#{det['id']} {det['type']}"
        cv2.putText(out, label, (x + 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_txt, 2)
        
    current_brightness = compute_brightness(gray, roi)
    # Mostramos la barra con TU VARIABLE objetivo
    draw_light_bar(out, current_brightness, TARGET_REAL_BRIGHTNESS)
    cv2.putText(out, f"Objetos: {len(detections)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
    
    return out

# =========================
# Gestión de Archivos JSON
# =========================
def clear_puntos_folder():
    os.makedirs(PUNTOS_DIR, exist_ok=True)
    for fname in os.listdir(PUNTOS_DIR):
        fpath = os.path.join(PUNTOS_DIR, fname)
        if os.path.isfile(fpath):
            try: os.remove(fpath)
            except OSError: pass

def save_points_json(image_name, detections):
    os.makedirs(PUNTOS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_name))[0]
    out_path = os.path.join(PUNTOS_DIR, base + "_puntos.json")
    data = {"imagen": image_name, "objetos": []}
    for det in detections:
        gc, p1, p2 = det.get("grip_center"), det.get("grip_p1"), det.get("grip_p2")
        if gc is None or p1 is None or p2 is None: continue
        obj = {
            "id": det.get("id"),
            "tipo": det.get("type"),
            "punto_central": {"x": int(gc[0]), "y": int(gc[1])},
            "punto_1": {"x": int(p1[0]), "y": int(p1[1])},
            "punto_2": {"x": int(p2[0]), "y": int(p2[1])},
            "es_tapon": det.get("has_cap", False) 
        }
        data["objetos"].append(obj)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Guardado JSON de puntos en: {out_path}")

# =========================
# Lógica común de Procesamiento
# =========================
def _procesar_y_mostrar(img, nombre_archivo):
    clear_puntos_folder()
    
    # === FILTRO CON TU VARIABLE ===
    img_corrected = apply_target_brightness_filter(img)
    
    gray = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY)
    detections, _ = get_detections(gray)
    save_points_json(nombre_archivo, detections)
    
    # Generar la imagen con todo dibujado
    out = process_frame(img) 
    
    # --- NUEVO: GUARDAR LA IMAGEN RESULTANTE ---
    base_name = os.path.basename(nombre_archivo)
    
    # Asegurar carpeta imagenes
    os.makedirs("imagenes", exist_ok=True)
    nombre_salida = os.path.join("imagenes", f"resultado_{base_name}")
    
    # Si la extensión no es amigable, forzamos png o jpg
    if not nombre_salida.lower().endswith(('.png', '.jpg', '.jpeg')):
        nombre_salida += ".png"
        
    cv2.imwrite(nombre_salida, out)
    print(f"🖼️ FOTO EXPORTADA: Guardada como '{nombre_salida}'")
    # --------------------------------------------
    
    cv2.imshow("FOTO (Target Real)", out)
    print("✅ Análisis completado. Pulsa cualquier tecla para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =========================
# Funciones Modos
# =========================
def analyze_image(path="image5.png"): 
    print(f"📂 Cargando imagen: {path}")
    img = cv2.imread(path)
    if img is None:
        print("❌ No se pudo abrir la imagen:", path)
        return
    _procesar_y_mostrar(img, path)

def analyze_camera_snapshot(cam_id=CAMERA_ID):
    print("🎥 Abriendo cámara para captura...")
    cap = open_cam(cam_id)
    if not cap.isOpened():
        print("❌ No se detectó ninguna cámara.")
        return
    print("⏳ Estabilizando cámara...")
    for _ in range(30): 
        cap.read()
        time.sleep(0.01)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("⚠️ Error al capturar imagen.")
        return
    print("📸 Foto capturada.")
    
    # Guardar en carpeta imagenes
    os.makedirs("imagenes", exist_ok=True)
    nombre_ficticio = os.path.join("imagenes", "captura_camara.png")
    
    # Guardamos la original por si acaso
    cv2.imwrite(nombre_ficticio, frame) 
    _procesar_y_mostrar(frame, nombre_ficticio)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("--- MASTER DEVELOPER VISION SYSTEM ---")
    print("1. [Enter] -> Modo FOTO")
    print("2. Escribe 'camara' -> Modo CÁMARA")
    
    modo = input("Opción: ").strip().lower()

    if modo == "camara":
        analyze_camera_snapshot(CAMERA_ID)
    else:
        # CAMBIA AQUÍ EL NOMBRE DE TU ARCHIVO
        # Buscamos la imagen una carpeta atrás respecto a este script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, "..", "captura_camara.png")
        analyze_image(img_path)
