#!/bin/bash
# Setup RunPod via SSH (sans SCP, en créant les fichiers directement)

SSH_USER="29chjnf1nryk78-64411a77@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "🚀 Setup RunPod via SSH (sans SCP)"
echo "==================================="
echo ""

# Méthode alternative : créer les fichiers essentiels directement via SSH
echo "🔧 Configuration sur RunPod..."

ssh -i "$SSH_KEY" ${SSH_USER} << 'ENDSSH'
set -e

echo "📁 Création de la structure..."
cd /workspace
mkdir -p finetuning_gilbert
cd finetuning_gilbert

# Créer la structure de base
mkdir -p outputs/models outputs/logs outputs/evaluations
mkdir -p data/processed data/raw data/test_sets
mkdir -p scripts src/model src/data src/training src/evaluation
mkdir -p config notebooks

echo "✅ Structure créée"

# Installer dépendances
echo ""
echo "📥 Installation des dépendances..."
pip install -q --upgrade pip
pip install -q transformers datasets accelerate librosa soundfile jiwer optimum[onnxruntime] torch torchaudio tqdm pyyaml

echo "✅ Dépendances installées"

# Vérifier GPU
echo ""
echo "🎮 Vérification GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "⚠️  GPU check"

echo ""
echo "✅ Setup de base terminé!"
echo ""
echo "💡 Deux options maintenant:"
echo "   1. Cloner votre repo depuis GitHub/GitLab (si disponible)"
echo "   2. Uploader les fichiers via JupyterLab (glisser-déposer)"
echo ""
echo "Pour JupyterLab, allez dans RunPod → Connect → Port 8888"
echo ""
ENDSSH

echo ""
echo "✅ Configuration de base terminée!"
echo ""
echo "📋 Prochaines étapes:"
echo ""
echo "Option 1 - Via Git (si votre repo est sur GitHub/GitLab):"
echo "   ssh -i ~/.ssh/id_ed25519 ${SSH_USER}"
echo "   cd /workspace/finetuning_gilbert"
echo "   git clone <votre-repo-url> ."
echo ""
echo "Option 2 - Via JupyterLab (recommandé):"
echo "   1. Aller sur RunPod → Connect → Port 8888 (Jupyter)"
echo "   2. Ouvrir JupyterLab"
echo "   3. Dans JupyterLab, aller dans /workspace/finetuning_gilbert"
echo "   4. Glisser-déposer vos fichiers depuis votre Mac"
echo "   5. Puis exécuter: bash setup_and_train.sh"
echo ""

