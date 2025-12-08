#!/bin/bash
# Script de setup automatique et lancement entraînement QAT

set -e  # Arrêter en cas d'erreur

echo "🚀 Setup et Entraînement QAT pour Whisper"
echo "=========================================="

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trouvé. Installez Python 3.8+"
    exit 1
fi

echo "✅ Python trouvé: $(python3 --version)"

# Vérifier GPU (optionnel)
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU détecté:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  Pas de GPU détecté. Training sera très lent sur CPU."
    read -p "Continuer quand même? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Créer environnement virtuel (optionnel mais recommandé)
if [ ! -d "venv" ]; then
    echo "📦 Création environnement virtuel..."
    python3 -m venv venv
fi

echo "📦 Activation environnement virtuel..."
source venv/bin/activate

# Installer dépendances
echo "📥 Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt
pip install optimum[onnxruntime]  # Pour quantization

echo "✅ Dépendances installées"

# Vérifier/télécharger dataset
echo "📊 Vérification des datasets..."
if [ ! -d "data/processed/common_voice_fr" ]; then
    echo "📥 Téléchargement Common Voice français..."
    python scripts/download_datasets.py --datasets common_voice --max_samples 60000
else
    echo "✅ Dataset trouvé: data/processed/common_voice_fr"
fi

# Vérifier que le dataset existe
if [ ! -d "data/processed/common_voice_fr" ]; then
    echo "❌ Dataset non trouvé. Vérifiez le téléchargement."
    exit 1
fi

# Lancer entraînement
echo ""
echo "🎯 Lancement de l'entraînement QAT..."
echo "   (Temps estimé: 2-4h sur GPU, 1-2 jours sur CPU)"
echo ""

python scripts/train_qat.py \
    --base_model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
    --train_data data/processed/common_voice_fr \
    --eval_data data/processed/common_voice_fr \
    --quantization_type int8 \
    --num_epochs 5 \
    --max_samples 60000 \
    --per_device_batch_size 8 \
    --output_dir outputs/models/whisper-qat-int8

echo ""
echo "✅ Entraînement terminé!"
echo "📁 Modèle sauvegardé dans: outputs/models/whisper-qat-int8/final"
echo ""
echo "💡 Prochaine étape: Conversion en modèle quantifié"
echo "   python scripts/convert_qat_to_quantized.py \\"
echo "     --model_path outputs/models/whisper-qat-int8/final \\"
echo "     --output_path outputs/models/whisper-qat-int8-quantized \\"
echo "     --quantization_type int8"

