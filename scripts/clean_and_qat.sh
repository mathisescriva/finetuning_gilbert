#!/bin/bash
# Script pour nettoyer le cache et lancer QAT avec dataset minimal

set -e

echo "🧹 Nettoyage cache HuggingFace..."
rm -rf /workspace/.hf_home/hub/* 2>/dev/null || true
rm -rf ~/.cache/huggingface/* 2>/dev/null || true

echo "🧹 Nettoyage cache pip..."
pip cache purge 2>/dev/null || true

echo "🧹 Nettoyage fichiers temporaires..."
rm -rf /tmp/* 2>/dev/null || true
rm -rf /workspace/tmp/* 2>/dev/null || true

echo ""
echo "💾 Espace disponible:"
df -h /workspace | tail -1

echo ""
echo "🚀 Lancement QAT avec dataset minimal (1000 échantillons)..."
echo ""

cd /workspace/finetuning_gilbert

python scripts/train_qat_simple.py \
  --base_model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --output_dir outputs/models/gilbert-whisper-qat-int8 \
  --max_samples 1000 \
  --num_epochs 3 \
  --batch_size 8 \
  --learning_rate 1e-5

