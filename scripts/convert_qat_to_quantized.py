#!/usr/bin/env python3
"""
Script pour convertir un modèle QAT (Quantization-Aware Training) 
en modèle quantifié réel (int8/int4) pour inférence.
"""

import argparse
import torch
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import sys

sys.path.append(str(Path(__file__).parent.parent))


def convert_to_onnx_quantized(model_path: str, output_path: str, quantization_type: str = "int8"):
    """
    Convertit un modèle PyTorch en ONNX quantifié.
    
    Args:
        model_path: Chemin vers modèle QAT PyTorch
        output_path: Chemin de sortie pour modèle quantifié
        quantization_type: "int8" ou "int4"
    """
    print(f"📦 Conversion {model_path} → ONNX quantifié ({quantization_type})...")
    
    # Charger modèle
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    
    # Configuration quantization
    if quantization_type == "int8":
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    elif quantization_type == "int4":
        # Int4 nécessite config custom (moins supporté)
        print("⚠️  Int4 nécessite configuration custom")
        print("   Utilisation int8 avec calibration avancée")
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=True)
    else:
        raise ValueError(f"Quantization type non supporté: {quantization_type}")
    
    # Quantifier
    print("🔧 Quantification...")
    quantizer = ORTQuantizer.from_pretrained(model, processor=processor)
    
    # Exporter modèle quantifié
    quantizer.export(
        onnx_model_path=str(Path(output_path) / "model.onnx"),
        onnx_quantized_model_output_path=str(Path(output_path) / "model_quantized.onnx"),
    )
    
    # Sauvegarder aussi le processor
    processor.save_pretrained(output_path)
    
    print(f"✅ Modèle quantifié sauvegardé dans {output_path}")


def convert_to_pytorch_quantized(model_path: str, output_path: str, quantization_type: str = "int8"):
    """
    Convertit un modèle QAT en PyTorch quantifié natif.
    
    Args:
        model_path: Chemin vers modèle QAT
        output_path: Chemin de sortie
        quantization_type: "int8" ou "int4"
    """
    print(f"📦 Conversion {model_path} → PyTorch quantifié ({quantization_type})...")
    
    # Charger modèle
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    
    try:
        from torch.quantization import convert, get_default_qconfig
        
        # Configuration quantization
        if quantization_type == "int8":
            qconfig = get_default_qconfig('fbgemm')  # ou 'qnnpack' pour CPU
        else:
            # Int4 nécessite custom quantizer (plus complexe)
            print("⚠️  Int4 PyTorch natif nécessite implémentation custom")
            print("   Utilisation int8 à la place")
            qconfig = get_default_qconfig('fbgemm')
        
        # Convertir (modifie le modèle in-place)
        model.qconfig = qconfig
        quantized_model = convert(model, inplace=False)
        
        # Sauvegarder
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(quantized_model.state_dict(), output_path / "pytorch_model.bin")
        
        # Sauvegarder config
        model.config.save_pretrained(output_path)
        
        print(f"✅ Modèle PyTorch quantifié sauvegardé dans {output_path}")
    
    except Exception as e:
        print(f"❌ Erreur conversion PyTorch: {e}")
        print("💡 Recommandation: Utiliser ONNX (plus supporté)")
        convert_to_onnx_quantized(model_path, output_path, quantization_type)


def main():
    parser = argparse.ArgumentParser(description="Convertir modèle QAT en quantifié")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Chemin vers modèle QAT entraîné",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Chemin de sortie pour modèle quantifié",
    )
    parser.add_argument(
        "--quantization_type",
        type=str,
        choices=["int8", "int4"],
        default="int8",
        help="Type de quantization",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "pytorch", "both"],
        default="onnx",
        help="Format de sortie (ONNX recommandé)",
    )
    
    args = parser.parse_args()
    
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    
    if args.format in ["onnx", "both"]:
        convert_to_onnx_quantized(
            args.model_path,
            args.output_path,
            args.quantization_type,
        )
    
    if args.format in ["pytorch", "both"]:
        pytorch_output = Path(args.output_path) / "pytorch"
        pytorch_output.mkdir(exist_ok=True)
        convert_to_pytorch_quantized(
            args.model_path,
            str(pytorch_output),
            args.quantization_type,
        )
    
    print(f"\n{'='*60}")
    print("✅ CONVERSION TERMINÉE")
    print(f"{'='*60}")
    print(f"\n💡 Modèle quantifié prêt pour évaluation et déploiement")


if __name__ == "__main__":
    main()

