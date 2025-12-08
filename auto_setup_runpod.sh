#!/bin/bash
# Script automatique de setup complet sur RunPod via SSH

set -e

POD_ID="${1:-m3djlqfzljissp-64411a7a}"

echo "🚀 Setup automatique sur RunPod"
echo "Pod ID: $POD_ID"
echo ""

# Uploader le projet
echo "📤 Upload du projet..."
cd /Users/mathisescriva/CascadeProjects/finetuning_gilbert
tar -czf /tmp/finetuning_gilbert.tar.gz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='outputs' --exclude='data/raw' --exclude='.DS_Store' . 2>/dev/null || true

# Upload via SSH
echo "📤 Upload vers RunPod..."
scp -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no /tmp/finetuning_gilbert.tar.gz ${POD_ID}@ssh.runpod.io:/workspace/ 2>&1 || {
    echo "⚠️  Upload échoué, tentative alternative..."
}

# Exécuter le setup sur RunPod
echo "🔧 Configuration sur RunPod..."
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no ${POD_ID}@ssh.runpod.io << 'ENDSSH'
set -e

echo "📦 Extraction du projet..."
cd /workspace
if [ -f finetuning_gilbert.tar.gz ]; then
    tar -xzf finetuning_gilbert.tar.gz 2>/dev/null || true
    rm -f finetuning_gilbert.tar.gz
fi

# Créer répertoire si nécessaire
mkdir -p finetuning_gilbert
cd finetuning_gilbert

echo "✅ Répertoire créé: $(pwd)"

# Vérifier GPU
echo ""
echo "🎮 Vérification GPU..."
nvidia-smi || echo "⚠️  GPU non détecté"

# Installer dépendances
echo ""
echo "📥 Installation des dépendances..."
pip install -q transformers datasets accelerate librosa soundfile jiwer optimum[onnxruntime] torch torchaudio tqdm pyyaml || pip3 install -q transformers datasets accelerate librosa soundfile jiwer optimum[onnxruntime] torch torchaudio tqdm pyyaml

echo "✅ Dépendances installées"

# Créer structure de répertoires
mkdir -p outputs/models outputs/logs outputs/evaluations data/processed

echo ""
echo "✅ Setup terminé!"
echo "📁 Projet dans: /workspace/finetuning_gilbert"
echo ""
echo "🚀 Prochaines étapes:"
echo "   cd /workspace/finetuning_gilbert"
echo "   bash setup_and_train.sh"
echo ""
ENDSSH

echo ""
echo "✅ Upload terminé!"
echo ""
echo "💡 Pour lancer l'entraînement, connectez-vous:"
echo "   ssh -i ~/.ssh/id_ed25519 ${POD_ID}@ssh.runpod.io"
echo "   cd /workspace/finetuning_gilbert"
echo "   bash setup_and_train.sh"

