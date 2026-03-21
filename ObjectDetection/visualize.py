"""
visualize.py — Run best.onnx on one image and draw detection boxes.
Usage:
  python visualize.py                          # random image from dataset
  python visualize.py --image path/to/img.jpg  # specific image
  python visualize.py --conf 0.25              # higher confidence threshold
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# Re-use helpers from run.py
from run import letterbox, xywh2xyxy, nms, yolo_postprocess, wbf, WBF_IOU, TTA_SCALES, TTA_FLIPS

ROOT = Path(__file__).resolve().parent
YOLO_ONNX = ROOT / "best.onnx"
ANNO_FILE = ROOT / "dataset" / "train" / "annotations.json"
IMG_DIR = ROOT / "dataset" / "train" / "images"
OUT_DIR = ROOT / "viz_output"


def load_category_names():
    with open(ANNO_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return {c["id"]: c["name"] for c in coco["categories"]}


# Distinct colours for categories (cycle through a fixed palette)
PALETTE = [
    "#FF3333", "#33FF57", "#3357FF", "#FF33A8", "#FF8C00",
    "#00CED1", "#9400D3", "#32CD32", "#FF6347", "#1E90FF",
    "#FFD700", "#ADFF2F", "#FF69B4", "#00FA9A", "#DC143C",
]


def cat_color(cat_id: int) -> str:
    return PALETTE[cat_id % len(PALETTE)]


def draw_detections(img: Image.Image, dets: list, cat_names: dict,
                    max_label_len: int = 30) -> Image.Image:
    draw = ImageDraw.Draw(img, "RGBA")

    # Try to load a small font; fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
        font_small = ImageFont.truetype("arial.ttf", size=11)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    for x1, y1, x2, y2, cls_id, conf in dets:
        color = cat_color(cls_id)
        name = cat_names.get(cls_id, f"cls_{cls_id}")
        if len(name) > max_label_len:
            name = name[:max_label_len - 1] + "…"
        label = f"{name} {conf:.2f}"

        # Semi-transparent fill
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 30))

        # Label background
        bbox = draw.textbbox((x1, y1), label, font=font)
        lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, y1 - lh - 4)
        draw.rectangle([x1, label_y, x1 + lw + 6, label_y + lh + 4],
                       fill=(r, g, b, 200))
        draw.text((x1 + 3, label_y + 2), label, fill="white", font=font)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image (default: random from dataset)")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Confidence threshold (default: 0.15)")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable TTA (faster, single-scale)")
    args = parser.parse_args()

    # Pick image
    if args.image:
        img_path = Path(args.image)
    else:
        images = sorted(IMG_DIR.glob("*.jpg"))
        img_path = random.choice(images)
    print(f"Image: {img_path.name}")

    # Load model
    print(f"Loading {YOLO_ONNX.name}...")
    session = ort.InferenceSession(str(YOLO_ONNX),
                                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    shape = session.get_inputs()[0].shape
    model_imgsz = shape[-1] if isinstance(shape[-1], int) and shape[-1] > 0 else 1280
    print(f"  Input size: {model_imgsz}px")

    # Load image
    pil_img = Image.open(str(img_path)).convert("RGB")
    orig_w, orig_h = pil_img.size
    img_np = np.array(pil_img)
    print(f"  Image size: {orig_w}×{orig_h}")

    # Inference
    tta_scales = [1.0] if args.no_tta else TTA_SCALES
    tta_flips  = [False] if args.no_tta else TTA_FLIPS

    all_dets = []
    for scale in tta_scales:
        for flip in tta_flips:
            size = max(32, int(model_imgsz * scale) // 32 * 32)
            inp = img_np[:, ::-1, :].copy() if flip else img_np
            padded, sc, pl, pt = letterbox(inp, size)
            blob = padded.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
            raw = session.run(None, {input_name: blob})
            dets = yolo_postprocess(raw[0], args.conf, 0.5, sc, pl, pt, orig_w, orig_h)
            if flip:
                dets = [(orig_w - x2, y1, orig_w - x1, y2, c, s)
                        for x1, y1, x2, y2, c, s in dets]
            all_dets.extend(dets)

    # Merge TTA with WBF
    if all_dets:
        boxes   = np.array([[d[0], d[1], d[2], d[3]] for d in all_dets], dtype=np.float32)
        scores  = np.array([d[5] for d in all_dets], dtype=np.float32)
        cls_ids = np.array([d[4] for d in all_dets], dtype=np.int32)
        boxes, scores, cls_ids = wbf(boxes, scores, cls_ids, WBF_IOU, args.conf)
        dets = [(float(boxes[i,0]), float(boxes[i,1]),
                 float(boxes[i,2]), float(boxes[i,3]),
                 int(cls_ids[i]), float(scores[i])) for i in range(len(scores))]
    else:
        dets = []

    print(f"  Detections: {len(dets)}")

    # Load categories and draw
    cat_names = load_category_names()
    result_img = draw_detections(pil_img.copy(), dets, cat_names)

    # Save
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{img_path.stem}_detections.jpg"
    result_img.save(str(out_path), quality=92)
    print(f"Saved → {out_path}")

    # Also print top detections
    if dets:
        print(f"\nTop detections (conf ≥ {args.conf}):")
        for x1, y1, x2, y2, cls_id, conf in sorted(dets, key=lambda d: -d[5])[:10]:
            name = cat_names.get(cls_id, f"cls_{cls_id}")
            print(f"  [{conf:.3f}] {name:40s}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")


if __name__ == "__main__":
    main()
