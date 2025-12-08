#!/usr/bin/env python3
"""
Script pour distillation et quantization du modèle fine-tuné.
"""

import argparse
import yaml
import torch
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import sys

sys.path.append(str(Path(__file__).parent.parent))


def quantize_to_int8(model_path: str, output_path: str):
    """
    Quantifie un modèle Whisper en int8 via ONNX.
    
    Args:
        model_path: Chemin vers modèle PyTorch
        output_path: Chemin de sortie pour modèle quantifié
    """
    print(f"Export ONNX du modèle {model_path}...")
    
    # Exporter vers ONNX
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # Config quantization
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    
    # Quantifier
    print("Quantification int8...")
    quantizer = ORTQuantizer.from_pretrained(model, processor=processor)
    
    # Générer dataset de calibration (optionnel, pour static quantization)
    # Pour dynamic quantization, pas nécessaire
    
    # Exporter modèle quantifié
    quantizer.export(
        onnx_model_path=str(Path(output_path) / "model.onnx"),
        onnx_quantized_model_output_path=str(Path(output_path) / "model_quantized.onnx"),
    )
    
    print(f"Modèle quantifié sauvegardé dans {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Distillation et quantization")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Chemin vers modèle fine-tuné",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Répertoire de sortie",
    )
    parser.add_argument(
        "--quantization_type",
        type=str,
        choices=["int8", "int4"],
        default="int8",
        help="Type de quantization",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Config modèle",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Quantization
    if args.quantization_type == "int8":
        quantize_to_int8(args.model_path, str(output_dir))
    else:
        print(f"Quantization {args.quantization_type} non encore implémentée")
        print("Utilisez int8 pour l'instant")


if __name__ == "__main__":
    main()

