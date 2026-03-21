#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# GCP VM Setup Script for NorgesGruppen Object Detection Training
# Run this on a fresh GCP VM with GPU (recommended: 1x A100 40GB or 1x L4 24GB)
#
# Recommended VM: g2-standard-8 (1x L4) or a2-highgpu-1g (1x A100)
# OS: Ubuntu 22.04 LTS with Deep Learning VM image (comes with CUDA + drivers)
# Disk: 200 GB SSD
# ──────────────────────────────────────────────────────────────────────────────
set -e

echo "=== NorgesGruppen Object Detection — GCP Setup ==="

# ── 1. System packages ──────────────────────────────────────────────────────
sudo apt-get update && sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    unzip htop tmux tree

# ── 2. Create project directory ─────────────────────────────────────────────
mkdir -p ~/objdet && cd ~/objdet

# ── 3. Python venv (match sandbox Python 3.11) ─────────────────────────────
python3.11 -m venv .venv
source .venv/bin/activate

# ── 4. Install packages (pinned to EXACT sandbox versions) ─────────────────
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics==8.1.0
pip install timm==0.9.12
pip install numpy==1.26.4
pip install onnx onnxruntime-gpu

# ── 5. Verify GPU ──────────────────────────────────────────────────────────
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "=== Setup complete! ==="
echo "Next steps:"
echo "  1. Upload your data: gcloud compute scp --recurse LOCAL_DIR VM_NAME:~/objdet/"
echo "  2. Or use gsutil: gsutil cp gs://YOUR_BUCKET/*.zip ~/objdet/"
echo "  3. Run: source .venv/bin/activate && python train_v3.py"
echo ""
echo "Expected files in ~/objdet/:"
echo "  NM_NGD_coco_dataset.zip    (854 MB - training images + annotations)"
echo "  NM_NGD_product_images.zip  (60 MB - product reference images)"
echo "  prepare_dataset.py"
echo "  train_v3.py"
echo "  build_embeddings.py"
echo "  data.yaml"
