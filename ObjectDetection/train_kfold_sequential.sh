#!/bin/bash
# Train all 3 folds sequentially on a single VM.
# Run inside tmux so SSH disconnect doesn't kill it.
#
# Usage:
#   tmux new -s kfold
#   bash train_kfold_sequential.sh
#   Ctrl+B D   ← detach

set -e
cd /home/erikk/objdet
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "FOLD 0 / 3"
echo "========================================"
python train_kfold.py --fold 0 --k 3 --imgsz 1280 --batch 2 --epochs 200

echo "========================================"
echo "FOLD 1 / 3"
echo "========================================"
python train_kfold.py --fold 1 --k 3 --imgsz 1280 --batch 2 --epochs 200

echo "========================================"
echo "FOLD 2 / 3"
echo "========================================"
python train_kfold.py --fold 2 --k 3 --imgsz 1280 --batch 2 --epochs 200

echo "========================================"
echo "ALL FOLDS DONE — exporting FP16 ONNX"
echo "========================================"
python export_and_package.py --weights runs/detect/fold_0/weights/best.pt --imgsz 1280 --half --name fold_0.onnx
python export_and_package.py --weights runs/detect/fold_1/weights/best.pt --imgsz 1280 --half --name fold_1.onnx
python export_and_package.py --weights runs/detect/fold_2/weights/best.pt --imgsz 1280 --half --name fold_2.onnx

echo "========================================"
echo "DONE. Files ready:"
ls -lh /home/erikk/objdet/fold_*.onnx
echo "========================================"
