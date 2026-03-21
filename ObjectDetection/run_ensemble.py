"""
run_ensemble.py — Run multiple ONNX models and merge with WBF for ensemble predictions.
Uses all *.onnx files in the same directory as this script (fold_0.onnx ... fold_4.onnx).
Falls back to best.onnx if no fold_*.onnx files found.

Usage:
  python run_ensemble.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from run import (letterbox, xywh2xyxy, nms, yolo_postprocess,
                 wbf, TTA_SCALES, TTA_FLIPS,
                 CONF_THRESHOLD, WBF_IOU)

SCRIPT_DIR = Path(__file__).resolve().parent


def run_model_on_image(session, input_name, model_imgsz, img_np, orig_w, orig_h,
                       pil_img) -> list:
    """Run one ONNX model on one image with multi-scale TTA + flip."""
    all_dets = []
    for scale in TTA_SCALES:
        for flip in TTA_FLIPS:
            size = max(32, int(model_imgsz * scale) // 32 * 32)
            inp = img_np[:, ::-1, :].copy() if flip else img_np
            padded, sc, pl, pt = letterbox(inp, size)
            blob = padded.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
            raw = session.run(None, {input_name: blob})
            dets = yolo_postprocess(raw[0], CONF_THRESHOLD, 0.5, sc, pl, pt, orig_w, orig_h)
            if flip:
                dets = [(orig_w - x2, y1, orig_w - x1, y2, c, s)
                        for x1, y1, x2, y2, c, s in dets]
            all_dets.extend(dets)
    return all_dets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Auto-discover ONNX models: prefer fold_*.onnx, fall back to best.onnx
    onnx_files = sorted(SCRIPT_DIR.glob("fold_*.onnx"))
    if not onnx_files:
        onnx_files = [SCRIPT_DIR / "best.onnx"]
    print(f"Ensemble: {len(onnx_files)} model(s): {[f.name for f in onnx_files]}")

    sessions = []
    for onnx_path in onnx_files:
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name = sess.get_inputs()[0].name
        shape = sess.get_inputs()[0].shape
        imgsz = shape[-1] if isinstance(shape[-1], int) and shape[-1] > 0 else 1280
        sessions.append((sess, input_name, imgsz))
        print(f"  Loaded {onnx_path.name} (imgsz={imgsz})")

    predictions = []
    input_dir = Path(args.input)
    output_path = Path(args.output)

    for img_path in sorted(input_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        image_id = int(img_path.stem.split("_")[-1])
        print(f"Processing {img_path.name} (image_id={image_id})")

        pil_img = Image.open(str(img_path)).convert("RGB")
        orig_w, orig_h = pil_img.size
        img_np = np.array(pil_img)

        # Collect detections from all models
        all_dets = []
        for sess, input_name, imgsz in sessions:
            dets = run_model_on_image(sess, input_name, imgsz, img_np, orig_w, orig_h, pil_img)
            all_dets.extend(dets)

        # Merge all detections with WBF
        if all_dets:
            boxes   = np.array([[d[0], d[1], d[2], d[3]] for d in all_dets], dtype=np.float32)
            scores  = np.array([d[5] for d in all_dets], dtype=np.float32)
            cls_ids = np.array([d[4] for d in all_dets], dtype=np.int32)
            boxes_w, scores_w, cls_ids_w = wbf(boxes, scores, cls_ids, WBF_IOU, CONF_THRESHOLD)
            dets = [(float(boxes_w[i, 0]), float(boxes_w[i, 1]),
                     float(boxes_w[i, 2]), float(boxes_w[i, 3]),
                     int(cls_ids_w[i]), float(scores_w[i]))
                    for i in range(len(scores_w))]
        else:
            dets = []

        for x1, y1, x2, y2, cat_id, conf in dets:
            predictions.append({
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [round(x1, 1), round(y1, 1),
                         round(x2 - x1, 1), round(y2 - y1, 1)],
                "score": round(conf, 3),
            })

        print(f"  detections: {len(dets)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
