"""
export_and_package.py — Export trained model to ONNX and build submission.zip.
Run this after training completes on GCP.

Usage:
  python export_and_package.py                              # auto-detect best.pt
  python export_and_package.py --weights runs/detect/grocery_v3/weights/best.pt
  python export_and_package.py --weights best.pt --imgsz 1280
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path




ROOT = Path(__file__).resolve().parent


def export_yolo_onnx(weights: Path, imgsz: int, half: bool = False) -> Path:
    """Export YOLO weights to ONNX format.
    half=True exports FP16 (~130MB vs 261MB) — use for multi-model ensembles.
    """
    from ultralytics import YOLO

    out_path = ROOT / "best.onnx"
    print(f"Exporting YOLO {weights} → ONNX (imgsz={imgsz}, half={half})...")
    model = YOLO(str(weights))
    model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True,
                 dynamic=True, half=half)

    exported = weights.with_suffix(".onnx")
    if exported != out_path:
        shutil.copy2(exported, out_path)
    print(f"  → {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return out_path


def create_submission_zip(yolo_onnx: Path) -> Path:
    """Package run.py + YOLO ONNX(es) into submission.zip.
    If fold_*.onnx files exist alongside yolo_onnx, packages all of them (max 3).
    """
    run_py = ROOT / "run.py"
    out_zip = ROOT / "submission.zip"

    # Collect weight files: all fold_*.onnx if present, else just best.onnx
    fold_files = sorted(ROOT.glob("fold_*.onnx"))
    weight_files = fold_files if fold_files else [yolo_onnx]

    files = {"run.py": run_py}
    for wf in weight_files:
        files[wf.name] = wf

    # Validate
    total_weight = 0
    weight_count = 0
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        sz = path.stat().st_size / 1024 / 1024
        ext = path.suffix.lower()
        if ext in (".onnx", ".pt", ".pth", ".npy", ".safetensors"):
            total_weight += sz
            weight_count += 1
        print(f"  {name:30s} {sz:7.1f} MB")

    print(f"\n  Weight files: {weight_count}/3 max")
    print(f"  Total weight size: {total_weight:.1f} MB / 420 MB max")

    if weight_count > 3:
        raise ValueError(f"Too many weight files: {weight_count} > 3")
    if total_weight > 420:
        raise ValueError(f"Weight files too large: {total_weight:.0f} MB > 420 MB")

    # Create zip
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(str(out_zip), "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in files.items():
            zf.write(str(path), name)

    print(f"\n  → {out_zip} ({out_zip.stat().st_size / 1024 / 1024:.1f} MB)")
    return out_zip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to YOLO best.pt (auto-detected if not specified)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="ONNX export image size (must match training imgsz)")
    parser.add_argument("--half", action="store_true",
                        help="Export FP16 ONNX (~130MB vs 261MB, for multi-model ensembles)")
    parser.add_argument("--name", type=str, default="best.onnx",
                        help="Output ONNX filename (default: best.onnx, use fold_0.onnx etc for ensemble)")
    args = parser.parse_args()

    # Auto-detect weights
    if args.weights:
        weights = Path(args.weights)
    else:
        candidates = [
            ROOT / "runs" / "detect" / "grocery_v3_full" / "weights" / "best.pt",
            ROOT / "runs" / "detect" / "grocery_v3" / "weights" / "best.pt",
            ROOT / "runs" / "detect" / "grocery_v2" / "weights" / "best.pt",
            ROOT / "best.pt",
        ]
        weights = None
        for c in candidates:
            if c.exists():
                weights = c
                break
        if weights is None:
            raise FileNotFoundError("No best.pt found. Specify --weights path")

    print(f"Using weights: {weights}")
    print(f"Export imgsz: {args.imgsz}")
    print()

    # Step 1: Export YOLO
    yolo_onnx = export_yolo_onnx(weights, args.imgsz, half=args.half)

    # Rename if custom name requested (for ensemble fold_N.onnx)
    if args.name != "best.onnx":
        dst = ROOT / args.name
        shutil.copy2(yolo_onnx, dst)
        yolo_onnx = dst
        print(f"  Saved as {dst.name}")

    # Step 2: Package (no EfficientNet — YOLO class predictions used directly)
    print("\nPackaging submission...")
    create_submission_zip(yolo_onnx)

    print("\n" + "=" * 60)
    print("Done! Upload submission.zip to the competition.")
    print("=" * 60)


if __name__ == "__main__":
    main()
