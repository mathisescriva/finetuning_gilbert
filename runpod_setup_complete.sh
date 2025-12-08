#!/bin/bash
# Script complet à exécuter dans VOTRE terminal pour setup RunPod

set -e

SSH_USER="29chjnf1nryk78-64411a77@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "🚀 Setup automatique sur RunPod"
echo "================================"
echo ""

# 1. Créer archive
echo "📦 Création de l'archive..."
cd /Users/mathisescriva/CascadeProjects/finetuning_gilbert
tar -czf /tmp/finetuning_gilbert.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='outputs' \
    --exclude='data/raw' \
    --exclude='.DS_Store' \
    --exclude='*.log' \
    .

echo "✅ Archive créée: $(ls -lh /tmp/finetuning_gilbert.tar.gz | awk '{print $5}')"

# 2. Upload
echo ""
echo "📤 Upload vers RunPod..."
scp -i "$SSH_KEY" /tmp/finetuning_gilbert.tar.gz ${SSH_USER}:/workspace/
echo "✅ Upload terminé"

# 3. Setup sur RunPod
echo ""
echo "🔧 Configuration sur RunPod..."
echo "   (Cela peut prendre quelques minutes)"
echo ""

ssh -i "$SSH_KEY" ${SSH_USER} << 'ENDSSH'
set -e

echo "📦 Extraction du projet..."
cd /workspace
if [ -f finetuning_gilbert.tar.gz ]; then
    tar -xzf finetuning_gilbert.tar.gz
    rm -f finetuning_gilbert.tar.gz
fi

# S'assurer que le répertoire existe
mkdir -p finetuning_gilbert
cd finetuning_gilbert

echo "✅ Projet extrait dans: $(pwd)"

# Vérifier GPU
echo ""
echo "🎮 Vérification GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "⚠️  GPU check failed"

# Installer dépendances
echo ""
echo "📥 Installation des dépendances..."
echo "   (Cela peut prendre 2-3 minutes)"
pip install -q --upgrade pip
pip install -q transformers datasets accelerate librosa soundfile jiwer optimum[onnxruntime] torch torchaudio tqdm pyyaml

# Créer structure de répertoires
mkdir -p outputs/models outputs/logs outputs/evaluations data/processed data/raw

# Vérifier que setup_and_train.sh est exécutable
chmod +x setup_and_train.sh scripts/*.sh 2>/dev/null || true

echo ""
echo "✅ ✅ ✅ SETUP TERMINÉ! ✅ ✅ ✅"
echo ""
echo "📁 Projet dans: /workspace/finetuning_gilbert"
echo "📋 Contenu:"
ls -la | head -15
echo ""
echo "🚀 Pour lancer l'entraînement QAT:"
echo "   cd /workspace/finetuning_gilbert"
echo "   bash setup_and_train.sh"
echo ""
ENDSSH

echo ""
echo "✅ ✅ ✅ SETUP COMPLET TERMINÉ! ✅ ✅ ✅"
echo ""
echo "💡 Pour lancer l'entraînement, connectez-vous:"
echo "   ssh -i ~/.ssh/id_ed25519 29chjnf1nryk78-64411a77@ssh.runpod.io"
echo "   cd /workspace/finetuning_gilbert"
echo "   bash setup_and_train.sh"
echo ""

