#!/bin/bash
# Script de nettoyage agressif du disque

echo "🧹 Nettoyage agressif du disque..."

# Supprimer fichiers temporaires
rm -rf /tmp/* 2>/dev/null
rm -rf /var/tmp/* 2>/dev/null

# Nettoyer cache pip
pip cache purge 2>/dev/null

# Nettoyer cache Python
rm -rf ~/.cache/pip 2>/dev/null
rm -rf ~/.cache/pypoetry 2>/dev/null

# Nettoyer cache HuggingFace (garder seulement les modèles essentiels)
# rm -rf ~/.cache/huggingface/hub/* 2>/dev/null  # ATTENTION: supprime tous les modèles en cache

# Nettoyer répertoires temporaires du projet
rm -rf /workspace/finetuning_gilbert/.git/ORIG_HEAD.lock 2>/dev/null
rm -rf /workspace/finetuning_gilbert/__pycache__ 2>/dev/null
rm -rf /workspace/finetuning_gilbert/**/__pycache__ 2>/dev/null

# Nettoyer outputs temporaires
rm -rf /workspace/finetuning_gilbert/outputs/models/whisper-ptq-int8/onnx/*.onnx_data 2>/dev/null

# Créer répertoire tmp sur /workspace
mkdir -p /workspace/tmp
export TMPDIR=/workspace/tmp
export TEMP=/workspace/tmp
export TMP=/workspace/tmp

echo "✅ Nettoyage terminé"
echo ""
echo "💾 Espace disque:"
df -h /workspace | tail -1

