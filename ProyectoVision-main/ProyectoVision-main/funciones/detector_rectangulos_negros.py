"""
Detect colored rectangles (yellow, brown, blue) and find the black rectangle center inside them.

Color ranges:
- Yellow: RGB(255, 246, 91) - hex #fff556
- Brown:  RGB(199, 151, 141) - hex #c9968a
- Blue:   RGB(179, 199, 255) - hex #b3c7ff
- Black:  RGB(0-50, 0-50, 0-50)

Usage:
  python detector_rectangulos_negros.py --image imagen.jpg --save resultado.png
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional


def detect_colored_rectangles(image_bgr: np.ndarray) -> Dict[str, Dict]:
    """Detect colored rectangles and black centers in the image."""
    h, w = image_bgr.shape[:2]
    
    # Color ranges in BGR (not HSV) with tolerance
    # Yellow: RGB(255, 246, 91) -> BGR(91, 246, 255)
    # Brown:  RGB(199, 151, 141) -> BGR(141, 151, 199)
    # Blue:   RGB(179, 199, 255) -> BGR(255, 199, 179)
    
    colors = {
        'amarillo': {'lower': np.array([50, 220, 230]), 'upper': np.array([130, 255, 255])},
        'marron': {'lower': np.array([110, 120, 170]), 'upper': np.array([170, 180, 230])},
        'azul': {'lower': np.array([230, 170, 150]), 'upper': np.array([255, 230, 210])}
    }
    
    results = {}
    
    for color_name, color_range in colors.items():
        lower = color_range['lower']
        upper = color_range['upper']
        mask = cv2.inRange(image_bgr, lower, upper)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            x, y, rect_w, rect_h = cv2.boundingRect(largest)
            area = cv2.contourArea(largest)
            
            if area > 500:  # Minimum area threshold
                results[color_name] = {
                    'x': x,
                    'y': y,
                    'width': rect_w,
                    'height': rect_h,
                    'area': area,
                    'center': (x + rect_w // 2, y + rect_h // 2)
                }
    
    return results


def find_black_rectangles_in_colored(image_bgr: np.ndarray, colored_rects: Dict) -> Dict[str, Dict]:
    """Find black rectangles inside each colored rectangle."""
    # Create mask for black areas (low brightness in HSV)
    bgr_black = cv2.inRange(image_bgr, np.array([0, 0, 0]), np.array([50, 50, 50]))
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_black = cv2.morphologyEx(bgr_black, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in black areas
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    black_rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Minimum area threshold
            x, y, rect_w, rect_h = cv2.boundingRect(cnt)
            center_x = x + rect_w // 2
            center_y = y + rect_h // 2
            black_rectangles.append({
                'x': x,
                'y': y,
                'width': rect_w,
                'height': rect_h,
                'area': area,
                'center': (center_x, center_y)
            })
    
    # Match black rectangles to colored rectangles
    result = {}
    
    for color_name, color_rect in colored_rects.items():
        cx, cy, cw, ch = color_rect['x'], color_rect['y'], color_rect['width'], color_rect['height']
        
        # Find black rectangles inside this colored rectangle
        blacks_inside = []
        for black_rect in black_rectangles:
            bx, by, bw, bh = black_rect['x'], black_rect['y'], black_rect['width'], black_rect['height']
            
            # Check if black rect is inside colored rect (with small margin)
            margin = 5
            if (bx > cx - margin and by > cy - margin and
                bx + bw < cx + cw + margin and by + bh < cy + ch + margin):
                blacks_inside.append(black_rect)
        
        # Get the largest black rectangle inside this colored rectangle
        if blacks_inside:
            largest_black = max(blacks_inside, key=lambda r: r['area'])
            result[color_name] = largest_black
    
    return result


def update_json_results(image_path: str, black_rects: Dict[str, Dict]):
    """Update the corresponding JSON file with detected black rectangles."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Determine points directory
    # Check if 'puntos' exists in current dir
    if os.path.isdir("puntos"):
        json_dir = "puntos"
    # Check if 'puntos' exists in parent dir (common if running from subfolder)
    elif os.path.isdir(os.path.join("..", "puntos")):
        json_dir = os.path.join("..", "puntos")
    else:
        # Default to local 'puntos'
        json_dir = "puntos"
        os.makedirs(json_dir, exist_ok=True)
        
    json_path = os.path.join(json_dir, f"{base_name}_puntos.json")
    
    data = {"imagen": image_path, "objetos": []}
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Error reading JSON {json_path}: {e}")
    
    # Filter out existing black rectangle entries from "objetos" if any exist from previous runs
    if "objetos" in data:
        data["objetos"] = [obj for obj in data["objetos"] 
                          if not obj.get("tipo", "").startswith("CAJA_NEGRA")]
    else:
        data["objetos"] = []
        
    # Initialize "contenedores" list
    data["contenedores"] = []
        
    # Add new black rectangles to contenedores
    container_id = 0
    for color, rect in black_rects.items():
        container_id += 1
        cx, cy = rect['center']
        x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
        
        # We use corners for p1 and p2 to represent the bounding box
        p1 = {"x": int(x), "y": int(y)}
        p2 = {"x": int(x + w), "y": int(y + h)}
        
        new_container = {
            "id": container_id,
            "tipo": f"CAJA_NEGRA_{color.upper()}",
            "punto_central": {"x": int(cx), "y": int(cy)},
            "punto_1": p1,
            "punto_2": p2
        }
        data["contenedores"].append(new_container)
        
    # Save JSON
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] JSON updated: {json_path}")
    except Exception as e:
        print(f"[ERROR] Error saving JSON: {e}")


def main():
    ap = argparse.ArgumentParser(description="Detect colored rectangles and black rectangle centers.")
    ap.add_argument("--image", type=str, required=True, help="Input image path")
    ap.add_argument("--save", type=str, default="", help="Output image path for visualization")
    args = ap.parse_args()
    
    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Cannot read image: {args.image}")
        return
    
    print(f"[INFO] Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Detect colored rectangles
    colored_rects = detect_colored_rectangles(img)
    print(f"[INFO] Detected {len(colored_rects)} colored rectangles:")
    for color, rect_info in colored_rects.items():
        print(f"  - {color}: center=({rect_info['center'][0]}, {rect_info['center'][1]}), "
              f"size={rect_info['width']}x{rect_info['height']}, area={rect_info['area']}")
    
    # Find black rectangles inside each colored rectangle
    black_rects = find_black_rectangles_in_colored(img, colored_rects)
    if black_rects:
        print(f"\n[INFO] Detected {len(black_rects)} black rectangles:")
        for color, black_rect in black_rects.items():
            cx, cy = black_rect['center']
            print(f"  - {color}: center=({cx}, {cy}), size={black_rect['width']}x{black_rect['height']}, area={black_rect['area']}")
    else:
        print("[WARN] No black rectangles found inside colored rectangles")
    
    # Visualize
    vis_img = img.copy()
    
    # Draw colored rectangles
    colors_bgr = {
        'amarillo': (91, 246, 255),   # BGR
        'marron': (141, 151, 199),
        'azul': (255, 199, 179)
    }
    
    for color_name, rect_info in colored_rects.items():
        x, y, w, h = rect_info['x'], rect_info['y'], rect_info['width'], rect_info['height']
        cx, cy = rect_info['center']
        bgr_color = colors_bgr.get(color_name, (0, 255, 0))
        
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), bgr_color, 3)
        cv2.circle(vis_img, (cx, cy), 6, bgr_color, -1)
        cv2.putText(vis_img, color_name, (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2, cv2.LINE_AA)
    
    # Draw black rectangle
    if black_rects:
        for color, black_rect in black_rects.items():
            x, y = black_rect['x'], black_rect['y']
            w, h = black_rect['width'], black_rect['height']
            cx, cy = black_rect['center']
            
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.circle(vis_img, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(vis_img, f"BLACK-{color} ({cx},{cy})", (cx+10, cy-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Rectangles Detection", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save if requested
    if args.save:
        os.makedirs("imagenes", exist_ok=True)
        save_path = args.save
        # If path doesn't have directory, put it in imagenes
        if not os.path.dirname(save_path):
            save_path = os.path.join("imagenes", save_path)
            
        cv2.imwrite(save_path, vis_img)
        print(f"[INFO] Visualization saved to {save_path}")

    # Update JSON results
    update_json_results(args.image, black_rects)


if __name__ == "__main__":
    main()
