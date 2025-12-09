#!/bin/bash
# Script d'entraînement QAT optimisé pour Vast.ai
# Focus: Performance/Frugalité/Vitesse

set -e

echo "🎯 Entraînement QAT - Modèle Propriétaire"
echo "=========================================="
echo ""

# Configuration
BASE_MODEL="bofenghuang/whisper-large-v3-distil-fr-v0.2"
QUANT_TYPE="int8"  # int8 ou int4
OUTPUT_DIR="outputs/models/gilbert-whisper-qat-${QUANT_TYPE}"
MAX_SAMPLES=60000  # ~500h de données (optimisé pour vitesse)
NUM_EPOCHS=5
# RTX 5090 peut gérer batch_size plus grand pour accélérer
BATCH_SIZE=16  # Optimisé pour RTX 5090 (peut même monter à 32 si VRAM suffit)
LEARNING_RATE=5e-6

# Vérifier qu'on est dans le bon répertoire
if [ ! -f "scripts/train_qat_optimized.py" ]; then
    echo "❌ Script train_qat_optimized.py non trouvé"
    echo "   Assurez-vous d'être dans le répertoire du projet"
    exit 1
fi

# Vérifier GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Pas de GPU détecté. Training sera très lent."
    read -p "Continuer quand même? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Configurer environnement
export HF_HOME=/workspace/.hf_home
export TRANSFORMERS_CACHE=/workspace/.hf_home
export TMPDIR=/workspace/tmp
export TEMP=/workspace/tmp
export TMP=/workspace/tmp

# Activer venv si disponible
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Vérifier espace disque
DISK_USAGE=$(df /workspace | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 85 ]; then
    echo "⚠️  Espace disque faible ($DISK_USAGE%), nettoyage..."
    bash scripts/free_disk_space.sh || echo "   Note: Script de nettoyage non disponible"
fi

# Vérifier datasets (on utilise maintenant streaming, pas besoin de télécharger)
echo "📊 Vérification datasets..."
echo "   Utilisation streaming (pas de téléchargement complet nécessaire)"
if [ ! -f "data/train.json" ]; then
    echo "   Dataset sera chargé en streaming depuis HuggingFace"
fi

# Déterminer train/eval data
if [ -f "data/train.json" ]; then
    TRAIN_DATA="data/train.json"
    EVAL_DATA="data/train.json"  # Utiliser même fichier pour eval (ou créer data/eval.json)
    echo "✅ Utilisation dataset local: $TRAIN_DATA"
else
    # Utiliser MLS (Multilingual LibriSpeech) - plus stable que Common Voice
    TRAIN_DATA="facebook/multilingual_librispeech"
    EVAL_DATA="facebook/multilingual_librispeech"
    TRAIN_DATA_CONFIG="french"
    EVAL_DATA_CONFIG="french"
    TRAIN_DATA_SPLIT="train"
    EVAL_DATA_SPLIT="dev"
    echo "✅ Utilisation dataset HuggingFace: $TRAIN_DATA (french)"
    echo "   Split train: $TRAIN_DATA_SPLIT"
    echo "   Split eval: $EVAL_DATA_SPLIT"
fi

# Créer répertoire de sortie
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "🚀 Démarrage entraînement QAT..."
echo "   Modèle de base: ${BASE_MODEL}"
echo "   Quantization: ${QUANT_TYPE}"
echo "   Output: ${OUTPUT_DIR}"
echo "   Échantillons: ${MAX_SAMPLES}"
echo "   Époques: ${NUM_EPOCHS}"
echo "   Batch size: ${BATCH_SIZE}"
echo ""

# Lancer entraînement
if [ -f "data/train.json" ]; then
    # Dataset local
    python scripts/train_qat_optimized.py \
        --base_model "${BASE_MODEL}" \
        --train_data "${TRAIN_DATA}" \
        --eval_data "${EVAL_DATA}" \
        --quantization_type "${QUANT_TYPE}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_epochs ${NUM_EPOCHS} \
        --max_samples ${MAX_SAMPLES} \
        --per_device_batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        2>&1 | tee "${OUTPUT_DIR}/training.log"
else
    # Dataset HuggingFace (MLS)
    python scripts/train_qat_optimized.py \
        --base_model "${BASE_MODEL}" \
        --train_data "${TRAIN_DATA}" \
        --eval_data "${EVAL_DATA}" \
        --quantization_type "${QUANT_TYPE}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_epochs ${NUM_EPOCHS} \
        --max_samples ${MAX_SAMPLES} \
        --per_device_batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        2>&1 | tee "${OUTPUT_DIR}/training.log"
fi

echo ""
echo "✅ Entraînement terminé !"
echo ""
echo "📋 Prochaines étapes:"
echo "   1. Convertir en modèle quantifié:"
echo "      python scripts/convert_qat_to_quantized.py \\"
echo "        --model_path ${OUTPUT_DIR}/final \\"
echo "        --output_path ${OUTPUT_DIR}-quantized \\"
echo "        --quantization_type ${QUANT_TYPE}"
echo ""
echo "   2. Benchmark performance:"
echo "      python scripts/benchmark_quantized.py \\"
echo "        --model_path ${OUTPUT_DIR}-quantized"
echo ""

