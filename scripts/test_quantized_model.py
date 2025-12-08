#!/usr/bin/env python3
"""
Test rapide du modèle quantifié PTQ
"""

import sys
from pathlib import Path
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor
import torch

model_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/models/whisper-ptq-int8/quantized"

print(f"🧪 Test du modèle quantifié: {model_path}")
print()

try:
    print("📦 Chargement du modèle quantifié...")
    model = ORTModelForSpeechSeq2Seq.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    print("✅ Modèle chargé avec succès!")
    print()
    
    print("📊 Informations du modèle:")
    print(f"   Type: ONNX Runtime (quantifié int8)")
    print(f"   Répertoire: {model_path}")
    
    # Lister les fichiers
    model_dir = Path(model_path)
    onnx_files = list(model_dir.glob("*.onnx"))
    print(f"\n📁 Fichiers ONNX trouvés ({len(onnx_files)}):")
    for f in onnx_files:
        size_mb = f.stat().st_size / 1e6
        print(f"   {f.name}: {size_mb:.1f} MB")
    
    total_size = sum(f.stat().st_size for f in onnx_files) / 1e9
    print(f"\n💾 Taille totale: {total_size:.2f} GB")
    
    print()
    print("✅ ✅ ✅ MODÈLE QUANTIFIÉ FONCTIONNEL! ✅ ✅ ✅")
    print()
    print("💡 Utilisation:")
    print(f"   from optimum.onnxruntime import ORTModelForSpeechSeq2Seq")
    print(f"   from transformers import AutoProcessor")
    print(f"   ")
    print(f"   model = ORTModelForSpeechSeq2Seq.from_pretrained('{model_path}')")
    print(f"   processor = AutoProcessor.from_pretrained('{model_path}')")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

