#!/bin/bash
# scripts/train.sh
# سكريبت التدريب

set -e

echo "🚀 Starting training..."

python -m notebooks.deep_learning_training \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --device cuda

echo "✅ Training complete!"