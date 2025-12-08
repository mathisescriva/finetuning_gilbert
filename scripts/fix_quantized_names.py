#!/usr/bin/env python3
"""
Renommer les fichiers ONNX quantifiés pour correspondre aux noms attendus par optimum
"""

import sys
from pathlib import Path
import shutil

model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/models/whisper-ptq-int8/quantized")

print(f"🔧 Renommage des fichiers ONNX quantifiés dans: {model_path}")
print()

# Renommer les fichiers
renames = {
    "encoder_model_quantized.onnx": "encoder_model.onnx",
    "decoder_model_quantized.onnx": "decoder_model.onnx",
}

for old_name, new_name in renames.items():
    old_path = model_path / old_name
    new_path = model_path / new_name
    
    if old_path.exists() and not new_path.exists():
        print(f"  {old_name} → {new_name}")
        old_path.rename(new_path)
        print(f"    ✅ Renommé")
    elif old_path.exists() and new_path.exists():
        print(f"  ⚠️  {new_name} existe déjà, suppression de {old_name}")
        old_path.unlink()
        print(f"    ✅ {old_name} supprimé")
    elif new_path.exists():
        print(f"  ✅ {new_name} existe déjà")
    else:
        print(f"  ❌ {old_name} non trouvé")

print()
print("✅ Renommage terminé!")

