"""
Convert COCO annotations to YOLO format and create dataset YAML.
Splits 248 images into ~90% train / ~10% val.
"""
import json
import random
import shutil
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
COCO_DIR = ROOT / "dataset" / "train"
ANN_FILE = COCO_DIR / "annotations.json"
IMG_DIR = COCO_DIR / "images"

YOLO_DIR = ROOT / "dataset_yolo"
YAML_PATH = ROOT / "data.yaml"

TRAIN_RATIO = 0.9
SEED = 42

# ── load COCO ────────────────────────────────────────────────────────────────
with open(ANN_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)

categories = coco["categories"]
images = coco["images"]
annotations = coco["annotations"]

# Build lookups
img_lookup = {img["id"]: img for img in images}
# Map image_id -> list of annotations
ann_by_image: dict[int, list] = {}
for ann in annotations:
    ann_by_image.setdefault(ann["image_id"], []).append(ann)

# Category names list (ordered by id)
cat_names = [c["name"] for c in sorted(categories, key=lambda c: c["id"])]
nc = len(cat_names)
print(f"Classes: {nc} | Images: {len(images)} | Annotations: {len(annotations)}")

# ── train/val split ──────────────────────────────────────────────────────────
random.seed(SEED)
img_ids = [img["id"] for img in images]
random.shuffle(img_ids)
split_idx = int(len(img_ids) * TRAIN_RATIO)
train_ids = set(img_ids[:split_idx])
val_ids = set(img_ids[split_idx:])
print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")

# ── create YOLO directory structure ──────────────────────────────────────────
for split in ("train", "val"):
    (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def coco_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] to YOLO [cx, cy, w, h] normalised."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


# ── convert and copy ────────────────────────────────────────────────────────
for img_id in img_ids:
    img_info = img_lookup[img_id]
    fname = img_info["file_name"]
    iw, ih = img_info["width"], img_info["height"]
    split = "train" if img_id in train_ids else "val"

    # Copy image
    src = IMG_DIR / fname
    dst = YOLO_DIR / "images" / split / fname
    if not dst.exists():
        shutil.copy2(src, dst)

    # Write label
    label_path = YOLO_DIR / "labels" / split / (Path(fname).stem + ".txt")
    lines = []
    for ann in ann_by_image.get(img_id, []):
        cls_id = ann["category_id"]
        cx, cy, nw, nh = coco_to_yolo(ann["bbox"], iw, ih)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")

# ── write YAML ───────────────────────────────────────────────────────────────
yaml_content = f"""# NorgesGruppen Object Detection - YOLOv8
path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val

nc: {nc}
names:
"""
for i, name in enumerate(cat_names):
    yaml_content += f"  {i}: \"{name}\"\n"

YAML_PATH.write_text(yaml_content, encoding="utf-8")
print(f"Dataset YAML written to {YAML_PATH}")
print("Done!")
