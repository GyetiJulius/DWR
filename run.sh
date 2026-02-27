#!/usr/bin/env bash
set -e

echo "=== Environment ==="
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}')"
echo ""

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt
echo ""

echo "=== Running Phase 1 forward-pass tests ==="
python test_forward.py
