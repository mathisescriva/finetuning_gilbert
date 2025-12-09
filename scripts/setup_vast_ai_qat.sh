#!/bin/bash
# Setup automatique pour QAT sur Vast.ai
# Optimisé pour performance/frugalité/vitesse

set -e  # Arrêter en cas d'erreur

echo "🚀 Setup QAT sur Vast.ai"
echo "========================"
echo ""

# Vérifier qu'on est sur /workspace (Vast.ai)
if [ ! -d "/workspace" ]; then
    echo "⚠️  /workspace non trouvé. Création..."
    mkdir -p /workspace
fi

cd /workspace

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trouvé"
    exit 1
fi
echo "✅ Python: $(python3 --version)"

# Vérifier GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU détecté:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  Pas de GPU détecté. QAT sera très lent sur CPU."
fi

# Vérifier espace disque
echo ""
echo "💾 Espace disque:"
df -h /workspace | tail -1

DISK_USAGE=$(df /workspace | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 85 ]; then
    echo "⚠️  Espace disque faible ($DISK_USAGE%), nettoyage..."
    pip cache purge 2>/dev/null || true
    rm -rf /tmp/* 2>/dev/null || true
    rm -rf ~/.cache/pip 2>/dev/null || true
fi

# Utiliser environnement conda si disponible
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✅ Utilisation conda: $CONDA_DEFAULT_ENV"
else
    echo "📦 Vérification environnement Python..."
    # Vérifier si venv existe
    if [ -d "venv" ]; then
        echo "✅ venv trouvé, activation..."
        source venv/bin/activate
    else
        echo "📦 Création venv..."
        python3 -m venv venv
        source venv/bin/activate
    fi
fi

# Installer dépendances
echo ""
echo "📥 Installation dépendances..."
pip install --upgrade pip --quiet
pip install --quiet \
    transformers>=4.35.0 \
    datasets>=2.14.0 \
    accelerate>=0.24.0 \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    jiwer>=3.0.0 \
    optimum[onnxruntime]>=1.14.0 \
    torch>=2.0.0 \
    torchaudio>=2.0.0 \
    tqdm>=4.65.0 \
    pyyaml>=6.0.0 \
    onnxruntime>=1.16.0

echo "✅ Dépendances installées"

# Configurer cache HuggingFace sur /workspace (plus d'espace)
echo ""
echo "⚙️  Configuration cache..."
export HF_HOME=/workspace/.hf_home
export TRANSFORMERS_CACHE=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home
mkdir -p /workspace/.hf_home

# Créer répertoires nécessaires
echo ""
echo "📁 Création structure..."
mkdir -p outputs/models
mkdir -p outputs/logs
mkdir -p outputs/evaluations
mkdir -p data/processed
mkdir -p /workspace/tmp

# Configurer TMPDIR
export TMPDIR=/workspace/tmp
export TEMP=/workspace/tmp
export TMP=/workspace/tmp

echo "✅ Structure créée"

# Vérifier PyTorch + CUDA
echo ""
echo "🔍 Vérification PyTorch..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "✅ Setup terminé !"
echo ""
echo "📋 Prochaines étapes:"
echo "   1. Cloner votre repo: git clone <repo-url> finetuning_gilbert"
echo "   2. cd finetuning_gilbert"
echo "   3. Lancer entraînement: bash scripts/train_qat_vast_ai.sh"
echo ""

