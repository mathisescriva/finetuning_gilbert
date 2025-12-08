#!/usr/bin/env python3
"""Vérifier ce qui existe dans les répertoires du modèle"""

from pathlib import Path
import sys

base_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/models/whisper-ptq-int8")

print(f"🔍 Vérification des fichiers dans: {base_path}")
print()

for subdir in ["onnx", "quantized"]:
    path = base_path / subdir
    if path.exists():
        print(f"📁 {subdir}/")
        files = list(path.rglob("*"))
        if files:
            for f in sorted(files):
                if f.is_file():
                    size_mb = f.stat().st_size / 1e6
                    print(f"   {f.name}: {size_mb:.1f} MB")
        else:
            print("   (vide)")
        print()
    else:
        print(f"❌ {subdir}/ n'existe pas")
        print()

