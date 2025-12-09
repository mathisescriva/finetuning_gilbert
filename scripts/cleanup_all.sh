#!/bin/bash
# Nettoyage agressif pour libérer de l'espace disque

echo "🧹 NETTOYAGE AGRESSIF DU DISQUE"
echo "================================"
echo ""

# 1. Cache HuggingFace
echo "📦 Nettoyage cache HuggingFace..."
rm -rf /workspace/.hf_home/hub/* 2>/dev/null || true
rm -rf /workspace/.hf_home/datasets/* 2>/dev/null || true
rm -rf ~/.cache/huggingface/* 2>/dev/null || true

# 2. Cache pip
echo "📦 Nettoyage cache pip..."
pip cache purge 2>/dev/null || true

# 3. Fichiers temporaires
echo "📦 Nettoyage fichiers temporaires..."
rm -rf /tmp/* 2>/dev/null || true
rm -rf /workspace/tmp/* 2>/dev/null || true
rm -rf /var/tmp/* 2>/dev/null || true

# 4. Anciens modèles téléchargés (garder seulement les plus récents)
echo "📦 Nettoyage anciens modèles..."
if [ -d "outputs/models" ]; then
    find outputs/models -name "*.onnx_data" -delete 2>/dev/null || true
    find outputs/models -name "*.pt" -mtime +1 -delete 2>/dev/null || true
fi

# 5. Logs anciens
echo "📦 Nettoyage logs..."
find outputs/logs -type f -mtime +1 -delete 2>/dev/null || true

# 6. Cache Python
echo "📦 Nettoyage cache Python..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "💾 Espace disponible:"
df -h /workspace | tail -1
echo ""

echo "✅ Nettoyage terminé !"

