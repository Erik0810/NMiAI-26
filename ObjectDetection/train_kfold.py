"""
train_kfold.py — Train one fold of a K-fold cross-validation ensemble.
Run this on K separate VMs simultaneously for maximum speed.

Each VM trains on (K-1)/K of the data and validates on 1/K.
At inference, all K models' predictions are merged with WBF.

Usage (on each VM, pass the fold index):
  python train_kfold.py --fold 0 --k 5 --imgsz 1280 --batch 2
  python train_kfold.py --fold 1 --k 5 --imgsz 1280 --batch 2
  ...
  python train_kfold.py --fold 4 --k 5 --imgsz 1280 --batch 2

After training, export on each VM:
  python export_and_package.py --weights runs/detect/fold_0/weights/best.pt --imgsz 1280

Then download all best.onnx files locally as fold_0.onnx ... fold_4.onnx
and run with run_ensemble.py.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def create_fold_yaml(root: Path, fold: int, k: int, seed: int = 42) -> Path:
    """Create a data.yaml for one fold: train on all-but-fold, val on fold."""
    ann_file = root / "dataset" / "train" / "annotations.json"
    with open(ann_file, "r") as f:
        coco = json.load(f)

    categories = coco["categories"]
    images = coco["images"]
    annotations = coco["annotations"]

    cat_names = {c["id"]: c["name"] for c in sorted(categories, key=lambda c: c["id"])}

    # Deterministic split
    img_ids = sorted([img["id"] for img in images])
    random.seed(seed)
    random.shuffle(img_ids)

    val_ids = set(img_ids[fold::k])  # every k-th image starting at fold
    train_ids = set(img_ids) - val_ids

    print(f"Fold {fold}/{k}: train={len(train_ids)} val={len(val_ids)}")

    yolo_dir = root / "dataset_yolo"
    fold_name = f"fold_{fold}"

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_out = yolo_dir / "images" / fold_name / split
        lbl_out = yolo_dir / "labels" / fold_name / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        # Symlink/copy from existing full dataset
        for img in images:
            if img["id"] not in ids:
                continue
            fname = img["file_name"]
            src_img = yolo_dir / "images" / "train" / fname
            src_lbl = yolo_dir / "labels" / "train" / (Path(fname).stem + ".txt")
            if not src_img.exists():
                src_img = yolo_dir / "images" / "val" / fname
                src_lbl = yolo_dir / "labels" / "val" / (Path(fname).stem + ".txt")

            dst_img = img_out / fname
            dst_lbl = lbl_out / (Path(fname).stem + ".txt")
            if not dst_img.exists() and src_img.exists():
                shutil.copy2(src_img, dst_img)
            if not dst_lbl.exists() and src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)

    yaml_path = root / f"data_fold_{fold}.yaml"
    lines = [
        f"path: {yolo_dir}",
        f"train: images/{fold_name}/train",
        f"val: images/{fold_name}/val",
        f"nc: {len(cat_names)}",
        "names:",
    ]
    for cid, cname in cat_names.items():
        lines.append(f'  {cid}: "{cname}"')
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0 to k-1)")
    parser.add_argument("--k", type=int, default=5, help="Total number of folds (default: 5)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size (default: 1280)")
    parser.add_argument("--batch", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs (default: 200)")
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="Base weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fold split")
    args = parser.parse_args()

    assert 0 <= args.fold < args.k, f"fold must be 0..{args.k-1}"

    yaml_path = create_fold_yaml(ROOT, args.fold, args.k, args.seed)

    run_name = f"fold_{args.fold}"
    last_ckpt = ROOT / "runs" / "detect" / run_name / "weights" / "last.pt"
    if last_ckpt.exists():
        print(f"Resuming fold {args.fold} from epoch checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        resume = True
    else:
        print(f"Starting fold {args.fold} fresh from {args.model}")
        model = YOLO(args.model)
        resume = False

    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=8,
        patience=40,
        save=True,
        save_period=25,
        amp=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.5,
        mosaic=1.0,
        close_mosaic=25,
        mixup=0.15,
        copy_paste=0.1,
        degrees=5.0,
        translate=0.2,
        scale=0.7,
        shear=2.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.3,
        max_det=300,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        project="runs/detect",
        name=run_name,
        exist_ok=True,
        resume=resume,
        verbose=True,
    )

    print(f"\nFold {args.fold} done! Best: runs/detect/{run_name}/weights/best.pt")


if __name__ == "__main__":
    main()
