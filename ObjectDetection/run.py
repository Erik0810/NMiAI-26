"""
run.py — Two-stage grocery detection + classification via ONNX Runtime.
Stage 1: YOLOv8x ONNX detects bounding boxes on shelf images (with multi-scale TTA).
Stage 2: EfficientNet-B0 ONNX embeddings match each crop to the nearest product.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# ── Constants ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
YOLO_ONNX = SCRIPT_DIR / "best.onnx"
EFFNET_ONNX = SCRIPT_DIR / "effnet_b0.onnx"
EMB_FILE = SCRIPT_DIR / "product_embeddings.npy"
MAP_FILE = SCRIPT_DIR / "product_mapping.json"

YOLO_IMG_SIZE = 1536
EFFNET_IMG_SIZE = 224
CONF_THRESHOLD = 0.07
IOU_THRESHOLD = 0.5
WBF_IOU = 0.55
TTA_SCALES = [1.0, 0.83, 1.17]
TTA_FLIPS = [False, True]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── YOLO helpers ─────────────────────────────────────────────────────────────
def letterbox(img: np.ndarray, new_shape: int = 640):
    """Resize image with letterbox padding (grey 114). Returns padded image and scale info."""
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))

    pad_h = new_shape - new_h
    pad_w = new_shape - new_w
    top = pad_h // 2
    left = pad_w // 2

    padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    padded[top : top + new_h, left : left + new_w] = resized

    return padded, scale, left, top


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert cx, cy, w, h → x1, y1, x2, y2."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Non-Maximum Suppression. boxes: (N, 4) xyxy, scores: (N,). Returns keep indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


def wbf(boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray,
        iou_thresh: float, conf_thresh: float) -> tuple:
    """
    Weighted Boxes Fusion — better than NMS for merging TTA predictions.
    Averages box coordinates weighted by confidence instead of discarding boxes.
    Returns (boxes, scores, cls_ids) as numpy arrays.
    """
    if len(boxes) == 0:
        return boxes, scores, cls_ids

    # Per-class WBF using offset trick (same class only merges together)
    max_coord = boxes.max() + 1
    offsets = cls_ids.astype(np.float32) * max_coord
    shifted = boxes + offsets[:, None]

    order = scores.argsort()[::-1]
    s_boxes  = shifted[order]
    s_scores = scores[order]
    s_cls    = cls_ids[order]
    o_boxes  = boxes[order]   # original coords for weighted avg

    # Each cluster: list of (box, score)
    clusters_box   = []   # weighted-sum box
    clusters_score = []   # sum of scores
    clusters_cls   = []   # majority class
    clusters_count = []   # how many boxes merged
    clusters_wbox  = []   # for weighted average accumulation

    def iou_1ton(box, cboxes):
        """IoU of one box against array of cluster representative boxes."""
        ix1 = np.maximum(box[0], cboxes[:, 0])
        iy1 = np.maximum(box[1], cboxes[:, 1])
        ix2 = np.minimum(box[2], cboxes[:, 2])
        iy2 = np.minimum(box[3], cboxes[:, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        a1 = (box[2]-box[0]) * (box[3]-box[1])
        a2 = (cboxes[:,2]-cboxes[:,0]) * (cboxes[:,3]-cboxes[:,1])
        return inter / (a1 + a2 - inter + 1e-7)

    cluster_reps = []  # shifted representative box per cluster

    for i in range(len(s_boxes)):
        if len(cluster_reps) == 0:
            cluster_reps.append(s_boxes[i])
            clusters_wbox.append(o_boxes[i] * s_scores[i])
            clusters_score.append(s_scores[i])
            clusters_cls.append(s_cls[i])
            clusters_count.append(1)
        else:
            reps = np.array(cluster_reps)
            ious = iou_1ton(s_boxes[i], reps)
            best = ious.argmax()
            if ious[best] > iou_thresh:
                # Merge into existing cluster
                clusters_wbox[best]  += o_boxes[i] * s_scores[i]
                clusters_score[best] += s_scores[i]
                clusters_count[best] += 1
                # Update representative to weighted avg
                avg = clusters_wbox[best] / clusters_score[best]
                off = clusters_cls[best] * max_coord
                cluster_reps[best] = avg + off
            else:
                cluster_reps.append(s_boxes[i])
                clusters_wbox.append(o_boxes[i] * s_scores[i])
                clusters_score.append(s_scores[i])
                clusters_cls.append(s_cls[i])
                clusters_count.append(1)

    # Build output: weighted average boxes, average confidence score
    out_boxes, out_scores, out_cls = [], [], []
    for i in range(len(clusters_wbox)):
        final_box = clusters_wbox[i] / clusters_score[i]
        # Average confidence = total score / number of boxes merged
        # Boxes seen by more TTA passes with high confidence score higher
        final_score = clusters_score[i] / clusters_count[i]
        if final_score >= conf_thresh:
            out_boxes.append(final_box)
            out_scores.append(final_score)
            out_cls.append(clusters_cls[i])

    if not out_boxes:
        empty = np.empty((0, 4), dtype=np.float32)
        return empty, np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    return (np.array(out_boxes, dtype=np.float32),
            np.array(out_scores, dtype=np.float32),
            np.array(out_cls, dtype=np.int32))


def yolo_postprocess(output: np.ndarray, conf_thresh: float, iou_thresh: float,
                     scale: float, pad_left: int, pad_top: int,
                     orig_w: int, orig_h: int):
    """
    Post-process YOLOv8 ONNX output.
    output shape: (1, 4+nc, 8400) → returns list of (x1, y1, x2, y2, cls_id, conf).
    """
    # (1, 360, 8400) → (8400, 360)
    preds = output[0].T

    # Split boxes and class scores
    boxes_cxcywh = preds[:, :4]                # (8400, 4)
    class_scores = preds[:, 4:]                # (8400, nc)

    # Max class score + class id per detection
    max_scores = class_scores.max(axis=1)      # (8400,)
    class_ids = class_scores.argmax(axis=1)    # (8400,)

    # Filter by confidence
    mask = max_scores > conf_thresh
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(max_scores) == 0:
        return []

    # Convert to xyxy
    boxes_xyxy = xywh2xyxy(boxes_cxcywh)

    # NMS per class (offset trick)
    max_coord = boxes_xyxy.max()
    offsets = class_ids.astype(np.float32) * (max_coord + 1)
    shifted = boxes_xyxy + offsets[:, None]
    keep = nms(shifted, max_scores, iou_thresh)

    boxes_xyxy = boxes_xyxy[keep]
    max_scores = max_scores[keep]
    class_ids = class_ids[keep]

    # Unscale: remove padding, then undo resize
    boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_left) / scale
    boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_top) / scale
    boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_left) / scale
    boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_top) / scale

    # Clip to image bounds
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    results = []
    for i in range(len(max_scores)):
        results.append((
            float(boxes_xyxy[i, 0]), float(boxes_xyxy[i, 1]),
            float(boxes_xyxy[i, 2]), float(boxes_xyxy[i, 3]),
            int(class_ids[i]), float(max_scores[i]),
        ))
    return results


# ── Classifier helpers ───────────────────────────────────────────────────────
def preprocess_crop(crop: Image.Image) -> np.ndarray:
    """Preprocess a crop for EfficientNet: resize, normalize, CHW, float32."""
    crop = crop.resize((EFFNET_IMG_SIZE, EFFNET_IMG_SIZE), Image.BILINEAR)
    arr = np.array(crop, dtype=np.float32) / 255.0        # (H, W, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)                           # (3, H, W)
    return arr


def classify_crops(crops: list, session: ort.InferenceSession,
                   ref_emb: np.ndarray, ref_cat_ids: np.ndarray,
                   unique_cats: np.ndarray) -> list:
    """Classify crops by cosine similarity to reference embeddings."""
    if not crops:
        return []

    input_name = session.get_inputs()[0].name
    batch_size = 32
    all_cat_ids = []

    for bi in range(0, len(crops), batch_size):
        batch_crops = crops[bi : bi + batch_size]
        batch_arr = np.stack([preprocess_crop(c) for c in batch_crops])  # (B, 3, 224, 224)
        feats = session.run(None, {input_name: batch_arr})[0]           # (B, 1280)

        # L2 normalise
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-7
        feats = feats / norms

        # Cosine similarity: (B, N_ref)
        sims = feats @ ref_emb.T

        for b in range(sims.shape[0]):
            best_cat = -1
            best_score = -1.0
            for cat in unique_cats:
                cat_mask = ref_cat_ids == cat
                cat_score = float(sims[b, cat_mask].max())
                if cat_score > best_score:
                    best_score = cat_score
                    best_cat = int(cat)
            all_cat_ids.append(best_cat)

    return all_cat_ids


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # ── Load YOLO ONNX (single best.onnx or ensemble of fold_*.onnx) ────
    onnx_files = sorted(SCRIPT_DIR.glob("fold_*.onnx"))
    if not onnx_files:
        onnx_files = [YOLO_ONNX]
    print(f"Models: {[f.name for f in onnx_files]}")

    sessions = []
    for onnx_path in onnx_files:
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name = sess.get_inputs()[0].name
        shape = sess.get_inputs()[0].shape
        imgsz = shape[-1] if isinstance(shape[-1], int) and shape[-1] > 0 else YOLO_IMG_SIZE
        sessions.append((sess, input_name, imgsz))
        print(f"  Loaded {onnx_path.name} (imgsz={imgsz})")

    yolo_session    = sessions[0][0]   # kept for compat
    yolo_input_name = sessions[0][1]
    model_imgsz     = sessions[0][2]

    # Classifier disabled: YOLO was trained with the correct 356 category_ids (0-355)
    # as class labels, so yolo_cls IS the category_id. A pretrained (non-fine-tuned)
    # EfficientNet using cosine similarity on generic ImageNet features cannot reliably
    # distinguish 356 similar grocery products and would overwrite correct YOLO predictions.
    use_classifier = False

    predictions = []

    for img_path in sorted(input_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        image_id = int(img_path.stem.split("_")[-1])
        print(f"Processing {img_path.name} (image_id={image_id})")

        pil_img = Image.open(str(img_path)).convert("RGB")
        orig_w, orig_h = pil_img.size
        img_np = np.array(pil_img)

        # ── YOLO inference with TTA ─────────────────────────────────
        all_dets = []

        for (sess, inp_name, sess_imgsz) in sessions:
            for tta_scale in TTA_SCALES:
                for tta_flip in TTA_FLIPS:
                    tta_size = max(32, int(sess_imgsz * tta_scale) // 32 * 32)
                    inp = img_np[:, ::-1, :].copy() if tta_flip else img_np
                    padded, sc, pl, pt = letterbox(inp, tta_size)
                    blob = padded.astype(np.float32) / 255.0
                    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
                    raw_out = sess.run(None, {inp_name: blob})
                    dets_tta = yolo_postprocess(
                        raw_out[0], CONF_THRESHOLD, IOU_THRESHOLD,
                        sc, pl, pt, orig_w, orig_h,
                    )
                    if tta_flip:
                        dets_tta = [
                            (orig_w - x2, y1, orig_w - x1, y2, cls, conf)
                            for x1, y1, x2, y2, cls, conf in dets_tta
                        ]
                    all_dets.extend(dets_tta)

        # Merge TTA detections with WBF (Weighted Boxes Fusion)
        if all_dets:
            boxes   = np.array([[d[0], d[1], d[2], d[3]] for d in all_dets], dtype=np.float32)
            scores  = np.array([d[5] for d in all_dets], dtype=np.float32)
            cls_ids = np.array([d[4] for d in all_dets], dtype=np.int32)
            boxes_w, scores_w, cls_ids_w = wbf(boxes, scores, cls_ids, WBF_IOU, CONF_THRESHOLD)
            dets = [
                (float(boxes_w[i, 0]), float(boxes_w[i, 1]),
                 float(boxes_w[i, 2]), float(boxes_w[i, 3]),
                 int(cls_ids_w[i]), float(scores_w[i]))
                for i in range(len(scores_w))
            ]
        else:
            dets = []

        # ── Build predictions ────────────────────────────────────────────
        # Use YOLO's class prediction directly — it was trained on the 356-category
        # grocery dataset so yolo_cls == category_id (0-355) is already correct.
        for idx, (x1, y1, x2, y2, yolo_cls, conf) in enumerate(dets):
            cat_id = yolo_cls
            predictions.append({
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [
                    round(x1, 1),
                    round(y1, 1),
                    round(x2 - x1, 1),
                    round(y2 - y1, 1),
                ],
                "score": round(conf, 3),
            })

        print(f"  detections: {len(dets)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
