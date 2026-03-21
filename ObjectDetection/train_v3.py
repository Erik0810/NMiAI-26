"""
train_v3.py — High-performance YOLOv8 training for NorgesGruppen competition.
Designed for GCP with A100 (40GB) or L4 (24GB) GPU.

Key improvements over v1/v2:
  1. YOLOv8x (largest) instead of YOLOv8m → better feature extraction
  2. imgsz=1280 instead of 640 → critical for shelf images with many small products
  3. Heavy augmentation: mosaic, mixup, copy-paste
  4. Longer training with cosine LR schedule
  5. Multi-scale training enabled
  6. AdamW optimizer for better convergence on small datasets
  7. Train on ALL 248 images (no val holdout) for final model, or 90/10 for monitoring

Usage:
  # Full training with validation monitoring:
  python train_v3.py

  # Final model (train on everything, no val):
  python train_v3.py --no-val

  # Resume from checkpoint:
  python train_v3.py --resume
"""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def create_full_dataset_yaml(root: Path) -> Path:
    """Create a data.yaml that uses ALL images for training (no val split)."""
    yaml_path = root / "data_full.yaml"
    yolo_dir = root / "dataset_yolo"

    # Merge val images into train for final training
    full_img_dir = yolo_dir / "images" / "full"
    full_lbl_dir = yolo_dir / "labels" / "full"
    full_img_dir.mkdir(parents=True, exist_ok=True)
    full_lbl_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        src_img = yolo_dir / "images" / split
        src_lbl = yolo_dir / "labels" / split
        for f in src_img.iterdir():
            dst = full_img_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
        for f in src_lbl.iterdir():
            dst = full_lbl_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)

    # Write yaml
    ann_file = root / "dataset" / "train" / "annotations.json"
    with open(ann_file, "r") as f:
        cats = json.load(f)["categories"]
    cat_names = {c["id"]: c["name"] for c in sorted(cats, key=lambda c: c["id"])}

    lines = [
        f"path: {yolo_dir}",
        "train: images/full",
        "val: images/full",  # same as train — just for ultralytics to not complain
        f"nc: {len(cat_names)}",
        "names:",
    ]
    for cid, cname in cat_names.items():
        lines.append(f'  {cid}: "{cname}"')

    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Created {yaml_path} with {sum(1 for _ in full_img_dir.iterdir())} images")
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-val", action="store_true",
                        help="Train on ALL images (no val split) for final model")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--weights", type=str, default=None,
                        help="Start from specific weights file (e.g. best.pt from a previous run)")
    parser.add_argument("--model", default="yolov8x.pt",
                        choices=["yolov8x.pt", "yolov8l.pt", "yolov8m.pt"],
                        help="Base model size (default: yolov8x)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Training image size (default: 1280)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Max epochs (default: 300)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size (-1 = auto-detect from VRAM)")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent

    # ── Dataset ──────────────────────────────────────────────────────────
    if args.no_val:
        data_yaml = create_full_dataset_yaml(ROOT)
        run_name = "grocery_v3_full"
    else:
        data_yaml = ROOT / "data.yaml"
        run_name = "grocery_v3"

    # ── Model ────────────────────────────────────────────────────────────
    if args.resume:
        ckpt = ROOT / "runs" / "detect" / run_name / "weights" / "last.pt"
        print(f"Resuming from {ckpt}")
        model = YOLO(str(ckpt))
    elif args.weights:
        print(f"Fine-tuning from {args.weights} (new run, imgsz may differ)")
        model = YOLO(args.weights)
    else:
        print(f"Starting fresh from {args.model}")
        model = YOLO(args.model)

    # ── Train ────────────────────────────────────────────────────────────
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,           # -1 = auto batch size from available VRAM
        device=0,
        workers=8,
        patience=50,               # longer patience — small dataset needs it
        save=True,
        save_period=25,            # checkpoint every 25 epochs
        val=True,
        amp=True,                  # mixed precision

        # ── Optimizer ────────────────────────────────────────────────────
        optimizer="AdamW",          # better for small datasets than SGD
        lr0=0.001,                  # lower initial LR for AdamW
        lrf=0.01,                   # cosine decay to lr0 * 0.01
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.5,

        # ── Augmentation — aggressive for small dataset ──────────────────
        mosaic=1.0,                 # mosaic augmentation (4 images combined)
        close_mosaic=30,            # disable mosaic last 30 epochs for fine-tuning
        mixup=0.15,                 # mixup augmentation
        copy_paste=0.1,             # copy-paste augmentation
        degrees=5.0,               # rotation ±5°
        translate=0.2,             # translation ±20%
        scale=0.7,                  # scale ±70% (allows big zoom in/out)
        shear=2.0,                 # shear ±2°
        perspective=0.0001,        # slight perspective
        flipud=0.0,                # no vertical flip (shelf images are oriented)
        fliplr=0.5,                # horizontal flip
        hsv_h=0.015,               # hue augmentation
        hsv_s=0.7,                 # saturation augmentation
        hsv_v=0.4,                 # value augmentation
        erasing=0.3,               # random erasing

        # ── Detection tuning ─────────────────────────────────────────────
        max_det=300,
        nbs=64,                    # nominal batch size for LR scaling
        box=7.5,                   # box loss weight
        cls=0.5,                   # classification loss weight
        dfl=1.5,                   # DFL loss weight

        # ── Output ───────────────────────────────────────────────────────
        project="runs/detect",
        name=run_name,
        exist_ok=True,
        resume=args.resume,
        verbose=True,
        plots=True,
    )

    print(f"\n{'='*60}")
    print(f"Training complete! Best weights: runs/detect/{run_name}/weights/best.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
