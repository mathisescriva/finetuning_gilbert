#!/bin/bash
# Script à exécuter depuis votre Mac pour uploader et setup sur RunPod

echo "📤 Upload et Setup sur RunPod"
echo ""
echo "Quelle est la commande SSH complète affichée dans RunPod?"
echo "Format: ssh POD-ID-XXXXX@ssh.runpod.io -i ~/.ssh/id_ed25519"
echo ""
read -p "Commande SSH complète: " SSH_CMD

if [ -z "$SSH_CMD" ]; then
    echo "❌ Commande SSH requise"
    exit 1
fi

# Extraire le user@host
SSH_TARGET=$(echo "$SSH_CMD" | sed 's/ssh //' | sed 's/ -i.*$//')

echo "📦 Création de l'archive..."
cd /Users/mathisescriva/CascadeProjects/finetuning_gilbert
tar -czf /tmp/finetuning_gilbert.tar.gz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='outputs' --exclude='data/raw' --exclude='.DS_Store' .

echo "📤 Upload vers RunPod..."
scp -i ~/.ssh/id_ed25519 /tmp/finetuning_gilbert.tar.gz ${SSH_TARGET}:/workspace/ || {
    echo "❌ Upload échoué"
    exit 1
}

echo "🔧 Configuration sur RunPod..."
ssh -i ~/.ssh/id_ed25519 ${SSH_TARGET} << 'ENDSSH'
cd /workspace
tar -xzf finetuning_gilbert.tar.gz 2>/dev/null || true
mkdir -p finetuning_gilbert
cd finetuning_gilbert
echo "✅ Projet dans: $(pwd)"
nvidia-smi
pip install -q transformers datasets accelerate librosa soundfile jiwer optimum[onnxruntime] torch torchaudio tqdm pyyaml
mkdir -p outputs/models outputs/logs outputs/evaluations data/processed
echo "✅ Setup terminé! Lancez: bash setup_and_train.sh"
ENDSSH

echo "✅ Terminé!"

