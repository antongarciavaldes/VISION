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


def detect_aruco_largest(image_bgr: np.ndarray, dict_name: str = "AUTO") -> Optional[Dict[str, Any]]:
    """Detect largest ArUco marker. Returns dict with id, center_px, W_px or None."""
    if image_bgr is None:
        return None
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

    best = None
    best_perim = -1.0

    for dn in dicts_to_try:
        aruco_dict = _get_aruco_dict(dn)
        for img in candidates:
            if hasattr(cv2.aruco, 'detectMarkers'):
                corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=params)
            else:
                detector = cv2.aruco.ArucoDetector(aruco_dict, params)
                corners, ids, _ = detector.detectMarkers(img)
            if ids is None or len(ids) == 0:
                continue
            for i in range(len(ids)):
                cs = corners[i].reshape(4, 2)
                peri = (np.linalg.norm(cs[1] - cs[0]) +
                        np.linalg.norm(cs[2] - cs[1]) +
                        np.linalg.norm(cs[3] - cs[2]) +
                        np.linalg.norm(cs[0] - cs[3]))
                if peri > best_perim:
                    sides = [
                        np.linalg.norm(cs[1] - cs[0]),
                        np.linalg.norm(cs[2] - cs[1]),
                        np.linalg.norm(cs[3] - cs[2]),
                        np.linalg.norm(cs[0] - cs[3]),
                    ]
                    W_px = float(np.mean(sides))
                    center = cs.mean(axis=0)
                    best = {
                        "id": int(ids[i][0]),
                        "center_px": {"x": float(center[0]), "y": float(center[1])},
                        "W_px": W_px,
                        "corners": cs.tolist()
                    }
                    best_perim = peri
        if best is not None:
            break
    return best


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
    det = detect_aruco_largest(frame, args.aruco_tamano)
    if det is None:
        print("[ERROR] No ArUco detected.")
        return

    mm_per_px = float(args.aruco_mm) / max(det["W_px"], 1e-6)
    print(f"[INFO] Detected ArUco ID={det['id']}, center=({det['center_px']['x']:.1f}, {det['center_px']['y']:.1f}), W_px={det['W_px']:.1f}, mm/px={mm_per_px:.4f}")

    # Process all objects
    results = {
        "imagen": args.image,
        "aruco": {
            "id": det["id"],
            "center_px": det["center_px"],
            "W_px": det["W_px"],
            "mm_per_px": mm_per_px
        },
        "objetos": []
    }

    for obj in points_data.get("objetos", []):
        obj_result = {"id": obj["id"]}
        
        # Convert each point in the object
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
        
        # Calculate distance between punto_1 and punto_2
        if punto_1_mm is not None and punto_2_mm is not None:
            dx = punto_2_mm["x_mm"] - punto_1_mm["x_mm"]
            dy = punto_2_mm["y_mm"] - punto_1_mm["y_mm"]
            distancia_mm = np.sqrt(dx**2 + dy**2)
            obj_result["distancia_punto1_punto2_mm"] = round(distancia_mm, 2)
        
        results["objetos"].append(obj_result)

    # Save output
    output_path = "resultados_mm.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Results saved to {output_path}")
    print(f"[INFO] Processed {len(results['objetos'])} objects")
    
    # Visualize results on image
    vis_img = frame.copy()
    # Draw ArUco detection
    cv2.circle(vis_img, (int(det["center_px"]["x"]), int(det["center_px"]["y"])), 8, (0, 255, 255), -1)
    cv2.putText(vis_img, f"ArUco ID={det['id']}", (int(det["center_px"]["x"])+15, int(det["center_px"]["y"])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Draw each object
    for obj_data in points_data.get("objetos", []):
        obj_id = obj_data["id"]
        # Find corresponding result
        obj_result = next((o for o in results["objetos"] if o["id"] == obj_id), None)
        if obj_result is None:
            continue
        
        # Draw central point
        if "punto_central" in obj_data:
            pc = obj_data["punto_central"]
            cv2.circle(vis_img, (pc["x"], pc["y"]), 6, (0, 255, 0), -1)
            cv2.putText(vis_img, f"ID{obj_id}", (pc["x"]+10, pc["y"]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
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
    vis_output = "resultados_visualizacion.png"
    cv2.imwrite(vis_output, vis_img)
    print(f"[INFO] Visualization saved to {vis_output}")
    
    cv2.imshow("ArUco Measurements", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
