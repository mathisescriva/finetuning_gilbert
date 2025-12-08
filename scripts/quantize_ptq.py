#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) pour Whisper.
Simple et rapide - pas besoin de données d'entraînement.
"""

import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from pathlib import Path
import os


def quantize_to_int8(model_name_or_path: str, output_path: str):
    """
    Quantifie un modèle Whisper en int8 avec PTQ.
    """
    print("🔧 Post-Training Quantization (PTQ) pour Whisper")
    print(f"📥 Modèle source: {model_name_or_path}")
    print(f"💾 Modèle de sortie: {output_path}")
    print()
    
    # Changer le cache HuggingFace vers /workspace (plus d'espace)
    os.environ["HF_HOME"] = "/workspace/.hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/.hf_home/hub"
    os.environ["HF_DATASETS_CACHE"] = "/workspace/.hf_home/datasets"
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Charger le modèle et le processeur
    print("📦 Chargement du modèle...")
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    # Sauvegarder le processeur
    processor.save_pretrained(output_path)
    
    print("✅ Modèle chargé")
    print(f"📊 Taille avant quantization: {sum(p.numel() * 4 for p in model.parameters()) / 1e9:.2f} GB (float32)")
    print()
    
    # Exporter vers ONNX d'abord
    print("🔄 Export ONNX...")
    onnx_model_path = output_path / "onnx"
    onnx_model_path.mkdir(exist_ok=True)
    
    try:
        # Exporter avec optimum
        print("  Exportation du modèle vers ONNX...")
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_name_or_path,
            export=True,
            use_cache=False,
        )
        onnx_model.save_pretrained(str(onnx_model_path))
        print("  ✅ Export ONNX réussi")
        
        # Quantifier
        print("🔢 Quantification int8...")
        quantizer = ORTQuantizer.from_pretrained(onnx_model_path)
        
        # Configuration quantization dynamic (pas besoin de calibration data)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
        
        print("  Application de la quantization...")
        quantizer.quantize(
            save_dir=str(output_path / "quantized"),
            quantization_config=qconfig,
        )
        
        print()
        print("✅ ✅ ✅ QUANTIZATION TERMINÉE! ✅ ✅ ✅")
        print(f"📁 Modèle quantifié dans: {output_path / 'quantized'}")
        print()
        print("💡 Utilisation:")
        print(f"   from optimum.onnxruntime import ORTModelForSpeechSeq2Seq")
        print(f"   model = ORTModelForSpeechSeq2Seq.from_pretrained('{output_path / 'quantized'}')")
        print()
        
        # Estimation taille
        if (output_path / "quantized").exists():
            total_size = sum(
                f.stat().st_size 
                for f in (output_path / "quantized").rglob("*") 
                if f.is_file()
            ) / 1e9
            original_size = sum(p.numel() * 4 for p in model.parameters()) / 1e9
            reduction = (1 - total_size / original_size) * 100
            print(f"📊 Taille après quantization: ~{total_size:.2f} GB (int8)")
            print(f"💾 Réduction: ~{reduction:.1f}%")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'export/quantization: {e}")
        print()
        print("💡 Alternative: Quantization PyTorch native (moins optimisé mais plus simple)")
        
        # Alternative: quantization PyTorch native
        try:
            print("\n🔄 Tentative avec quantization PyTorch native...")
            model_quantized = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            quantized_path = output_path / "quantized_pytorch"
            quantized_path.mkdir(exist_ok=True)
            model_quantized.save_pretrained(str(quantized_path))
            processor.save_pretrained(str(quantized_path))
            
            print(f"✅ Modèle quantifié PyTorch sauvegardé dans: {quantized_path}")
            print("   (Moins optimisé que ONNX mais fonctionne)")
            
        except Exception as e2:
            print(f"❌ Erreur avec PyTorch quantization: {e2}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Post-Training Quantization (PTQ) pour Whisper - Simple et rapide"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bofenghuang/whisper-large-v3-distil-fr-v0.2",
        help="Modèle HuggingFace à quantifier",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/models/whisper-ptq-int8",
        help="Répertoire de sortie",
    )
    
    args = parser.parse_args()
    
    quantize_to_int8(args.model, args.output)


if __name__ == "__main__":
    main()

