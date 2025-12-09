#!/bin/bash
# Test rapide du modèle quantifié PTQ

set -e

MODEL_PATH="outputs/models/gilbert-whisper-ptq-int8/quantized"

echo "🧪 TEST DU MODÈLE QUANTIFIÉ PTQ"
echo "================================"
echo ""

cd /workspace/finetuning_gilbert

# 1. Test de chargement
echo "📦 Test de chargement..."
python << 'PYEOF'
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor
from pathlib import Path
import numpy as np

model_path = "outputs/models/gilbert-whisper-ptq-int8/quantized"

print(f"Chargement depuis: {model_path}")

model = ORTModelForSpeechSeq2Seq.from_pretrained(
    model_path,
    encoder_file_name="encoder_model_quantized.onnx",
    decoder_file_name="decoder_model_quantized.onnx",
    use_cache=False,
)
processor = AutoProcessor.from_pretrained(model_path)

print("✅ Modèle chargé avec succès!")

# Taille
model_dir = Path(model_path)
onnx_files = list(model_dir.glob("*.onnx"))
total_size = sum(f.stat().st_size for f in onnx_files) / 1e9
print(f"\n💾 Taille: {total_size:.2f} GB")

# Test inférence rapide
print("\n🔍 Test inférence...")
dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 seconde
inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")

# Test génération (sans accès direct à session)
output = model.generate(**inputs, max_length=50, language="fr")
transcription = processor.batch_decode(output, skip_special_tokens=True)[0]

print(f"✅ Transcription test: '{transcription[:50]}...'")
print("\n✅ ✅ ✅ MODÈLE FONCTIONNEL! ✅ ✅ ✅")
PYEOF

echo ""
echo "📊 Comparaison avec modèle original..."
python scripts/benchmark_model.py \
  --model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --quantized "$MODEL_PATH" \
  --device cuda \
  --num_runs 5

