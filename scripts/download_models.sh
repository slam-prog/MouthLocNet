#!/bin/bash
# scripts/download_models.sh
# سكريبت تحميل النماذج

set -e

echo "📥 Downloading pretrained models..."

python -m models.download_pretrained

echo "✅ Models downloaded!"