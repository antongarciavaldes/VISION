"""
Batch measurement of pixel points to mm coordinates using ArUco reference.
Reads points from JSON, detects ArUco, calculates mm positions, outputs JSON.

Usage:
  python aruco_batch_measure.py --image image2.png --points-json puntos/image2_puntos.json --aruco-mm 50 --aruco-tamano AUTO
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np


def _get_aruco_dict(dict_name: str):
    name = dict_name.strip().upper()
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    return cv2.aruco.Dictionary_get(getattr(cv2.aruco, name))


def _make_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        p = cv2.aruco.DetectorParameters_create()
    else:
        p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 10
    p.adaptiveThreshConstant = 7
    p.minMarkerPerimeterRate = 0.01
    p.maxMarkerPerimeterRate = 4.0
    if hasattr(p, "cornerRefinementMethod"):
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


def detect_arucos(image_bgr: np.ndarray, dict_name: str = "AUTO") -> List[Dict[str, Any]]:
    """Detect all ArUco markers. Returns list of dicts with id, center_px, W_px."""
    if image_bgr is None:
        return []
    params = _make_detector_params()
    
    dicts_to_try = []
    if dict_name.strip().upper() == "AUTO":
        dicts_to_try = ["DICT_5X5_50", "DICT_5X5_100", "DICT_4X4_50", "DICT_4X4_100", "DICT_6X6_50"]
    else:
        dicts_to_try = [dict_name]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    candidates = [gray, image_bgr, gray_eq]

    detected_markers = []

    for dn in dicts_to_try:
        aruco_dict = _get_aruco_dict(dn)
        for img in candidates:
            if hasattr(cv2.aruco, 'detectMarkers'):
                corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=params)
            else:
                detector = cv2.aruco.ArucoDetector(aruco_dict, params)
                corners, ids, _ = detector.detectMarkers(img)
            
            if ids is not None and len(ids) > 0:
                # Found markers
                for i in range(len(ids)):
                    cs = corners[i].reshape(4, 2)
                    sides = [
                        np.linalg.norm(cs[1] - cs[0]),
                        np.linalg.norm(cs[2] - cs[1]),
                        np.linalg.norm(cs[3] - cs[2]),
                        np.linalg.norm(cs[0] - cs[3]),
                    ]
                    W_px = float(np.mean(sides))
                    center = cs.mean(axis=0)
                    detected_markers.append({
                        "id": int(ids[i][0]),
                        "center_px": {"x": float(center[0]), "y": float(center[1])},
                        "W_px": W_px,
                        "corners": cs.tolist()
                    })
                return detected_markers
    return []


def pixel_to_mm(px: float, py: float, marker_center_px: Dict, mm_per_px: float, 
                ref_x_mm: float = 0.0, ref_y_mm: float = 0.0,
                flip_x: bool = False, flip_y: bool = False) -> Dict[str, float]:
    """Convert pixel coords to mm relative to marker center + reference offset."""
    cx, cy = marker_center_px["x"], marker_center_px["y"]
    dx_px = px - cx
    dy_px = py - cy
    if flip_x:
        dx_px = -dx_px
    if flip_y:
        dy_px = -dy_px
    dx_mm = dx_px * mm_per_px
    dy_mm = dy_px * mm_per_px
    return {
        "x_mm": ref_x_mm + dx_mm,
        "y_mm": ref_y_mm + dy_mm
    }


def process_items(items_list: List[Dict], det: Dict, mm_per_px: float) -> List[Dict]:
    """Helper to process a list of items (objects or containers) and convert to mm."""
    processed = []
    for obj in items_list:
        obj_result = {"id": obj.get("id")}
        if "tipo" in obj:
            obj_result["tipo"] = obj["tipo"]
        
        punto_1_mm = None
        punto_2_mm = None
        
        for key in ["punto_central", "punto_1", "punto_2"]:
            if key in obj:
                pt = obj[key]
                mm_coords = pixel_to_mm(
                    pt["x"], pt["y"],
                    det["center_px"],
                    mm_per_px,
                    0.0, 0.0,
                    False, False
                )
                obj_result[key] = {
                    "x_px": pt["x"],
                    "y_px": pt["y"],
                    "x_mm": round(mm_coords["x_mm"], 2),
                    "y_mm": round(mm_coords["y_mm"], 2)
                }
                
                if key == "punto_1":
                    punto_1_mm = mm_coords
                elif key == "punto_2":
                    punto_2_mm = mm_coords
        
        if punto_1_mm and punto_2_mm:
            dx = punto_2_mm["x_mm"] - punto_1_mm["x_mm"]
            dy = punto_2_mm["y_mm"] - punto_1_mm["y_mm"]
            dist = np.sqrt(dx*dx + dy*dy)
            obj_result["distancia_punto1_punto2_mm"] = round(dist, 2)
            
        processed.append(obj_result)
    return processed


def main():
    ap = argparse.ArgumentParser(description="Batch convert pixel points to mm using ArUco reference.")
    ap.add_argument("--image", type=str, required=True, help="Input image path")
    ap.add_argument("--points-json", type=str, required=True, help="JSON with pixel points")
    ap.add_argument("--aruco-mm", type=float, default=50.0, help="ArUco marker side length in mm")
    ap.add_argument("--aruco-tamano", type=str, default="AUTO", help="ArUco dictionary (e.g., DICT_4X4_50, AUTO)")
    args = ap.parse_args()

    # Load image
    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[ERROR] Cannot read image: {args.image}")
        return

    # Load points JSON
    with open(args.points_json, "r", encoding="utf-8") as f:
        points_data = json.load(f)

    # Detect ArUco
    markers = detect_arucos(frame, args.aruco_tamano)
    if not markers:
        print("[ERROR] No ArUco detected.")
        return

    # Calculate average scale
    scales = []
    print(f"[INFO] Detected {len(markers)} ArUco markers:")
    for m in markers:
        scale = float(args.aruco_mm) / max(m["W_px"], 1e-6)
        scales.append(scale)
        print(f"  - ID={m['id']}, Center=({m['center_px']['x']:.1f}, {m['center_px']['y']:.1f}), W_px={m['W_px']:.1f}, mm/px={scale:.4f}")
    
    mm_per_px = float(np.mean(scales))
    print(f"[INFO] Using Average mm/px = {mm_per_px:.4f}")

    # Use the largest marker as the coordinate origin reference
    main_marker = max(markers, key=lambda m: m["W_px"])
    print(f"[INFO] Using Marker ID={main_marker['id']} as origin reference.")

    # Process all objects and containers
    results = {
        "imagen": args.image,
        "aruco": {
            "ids": [m["id"] for m in markers],
            "reference_id": main_marker["id"],
            "center_px": main_marker["center_px"],
            "avg_mm_per_px": mm_per_px
        },
        "objetos": process_items(points_data.get("objetos", []), main_marker, mm_per_px),
        "contenedores": process_items(points_data.get("contenedores", []), main_marker, mm_per_px)
    }

    # Save output
    output_dir = os.path.dirname(args.points_json)
    if not output_dir:
        output_dir = "puntos"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "resultados_mm.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Results saved to {output_path}")
    print(f"[INFO] Processed {len(results['objetos'])} objects and {len(results['contenedores'])} containers")
    
    # Visualize results on image
    vis_img = frame.copy()
    # Draw ArUco detections
    for m in markers:
        cx, cy = int(m["center_px"]["x"]), int(m["center_px"]["y"])
        cv2.circle(vis_img, (cx, cy), 8, (0, 255, 255), -1)
        cv2.putText(vis_img, f"ID={m['id']}", (cx+15, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        # Draw corners
        corners = np.array(m["corners"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [corners], True, (0, 255, 255), 2)
    
    # Draw each object and container
    for category in ["objetos", "contenedores"]:
        color_circle = (0, 255, 0) if category == "objetos" else (0, 165, 255) # Green for objects, Orange for containers
        
        for obj_data in points_data.get(category, []):
            obj_id = obj_data.get("id")
            # Find corresponding result
            obj_result = next((o for o in results[category] if o["id"] == obj_id), None)
            if obj_result is None:
                continue
            
            # Draw central point
            if "punto_central" in obj_data:
                pc = obj_data["punto_central"]
                cv2.circle(vis_img, (pc["x"], pc["y"]), 6, color_circle, -1)
                label = f"ID{obj_id}" if category == "objetos" else f"C{obj_id}"
                cv2.putText(vis_img, label, (pc["x"]+10, pc["y"]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_circle, 2, cv2.LINE_AA)
            
            # Draw line between punto_1 and punto_2
            if "punto_1" in obj_data and "punto_2" in obj_data:
                p1 = obj_data["punto_1"]
                p2 = obj_data["punto_2"]
                cv2.line(vis_img, (p1["x"], p1["y"]), (p2["x"], p2["y"]), (255, 0, 0), 2)
                cv2.circle(vis_img, (p1["x"], p1["y"]), 4, (255, 255, 0), -1)
                cv2.circle(vis_img, (p2["x"], p2["y"]), 4, (255, 255, 0), -1)
                
                # Draw distance text at midpoint
                if "distancia_punto1_punto2_mm" in obj_result:
                    mid_x = (p1["x"] + p2["x"]) // 2
                    mid_y = (p1["y"] + p2["y"]) // 2
                    dist_text = f"{obj_result['distancia_punto1_punto2_mm']:.1f}mm"
                    cv2.putText(vis_img, dist_text, (mid_x+5, mid_y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
    
    # Save and show visualization
    os.makedirs("imagenes", exist_ok=True)
    vis_output = os.path.join("imagenes", "resultados_visualizacion.png")
    cv2.imwrite(vis_output, vis_img)
    print(f"[INFO] Visualization saved to {vis_output}")
    
    cv2.imshow("ArUco Measurements", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
