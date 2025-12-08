#!/usr/bin/env python3
"""
Script pour évaluer un modèle QAT/quantifié sur les mêmes corpus que v0.2.
Permet comparaison directe avec baseline.
"""

import argparse
import json
import yaml
from pathlib import Path
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.evaluator import WhisperEvaluator


def load_quantized_model(model_path: str, model_type: str = "auto"):
    """
    Charge un modèle quantifié (détecte automatiquement le format).
    
    Args:
        model_path: Chemin vers modèle
        model_type: "auto", "onnx", "pytorch", "transformers"
    
    Returns:
        Modèle et processor
    """
    processor = WhisperProcessor.from_pretrained(model_path)
    
    # Détecter format
    if model_type == "auto":
        onnx_path = Path(model_path) / "model_quantized.onnx"
        if onnx_path.exists():
            model_type = "onnx"
        elif (Path(model_path) / "pytorch_model.bin").exists():
            model_type = "pytorch"
        else:
            model_type = "transformers"
    
    print(f"Chargement modèle {model_type} depuis {model_path}...")
    
    if model_type == "onnx":
        # Modèle ONNX quantifié
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            file_name="model_quantized.onnx",
        )
    elif model_type == "transformers":
        # Modèle PyTorch standard (peut être QAT préparé)
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
    else:
        # PyTorch quantifié natif (plus complexe à charger)
        print("⚠️  PyTorch quantifié natif nécessite chargement custom")
        print("   Utilisation modèle transformers standard")
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    return model, processor


def evaluate_on_corpus(model, processor, corpus_name: str, split: str = "test"):
    """
    Évalue sur un corpus spécifique (comme dans la model card).
    
    Corpus supportés:
    - community-v2/dev_data
    - mtedx
    - zaion5
    - zaion6
    """
    print(f"\n📊 Évaluation sur {corpus_name}...")
    
    # Mapping des corpus (à adapter selon disponibilité)
    corpus_mapping = {
        "community-v2": "mozilla-foundation/common_voice_17_0",
        "mtedx": "facebook/mtedx",
        # zaion5/6 sont des datasets internes, à adapter
    }
    
    try:
        from datasets import load_dataset
        
        if corpus_name in corpus_mapping:
            dataset_name = corpus_mapping[corpus_name]
            dataset = load_dataset(dataset_name, "fr", split=split)
        else:
            # Essayer de charger directement
            dataset = load_dataset(corpus_name, split=split)
        
        # Préparer données pour évaluation
        # (simplifié, à adapter selon format exact du corpus)
        print(f"   Dataset chargé: {len(dataset)} échantillons")
        
        # Ici, on devrait adapter selon le format exact de chaque corpus
        # Pour l'instant, structure générique
        return {
            "corpus": corpus_name,
            "num_samples": len(dataset),
            "status": "loaded",
            "note": "Évaluation à implémenter selon format corpus",
        }
    
    except Exception as e:
        print(f"   ⚠️  Corpus {corpus_name} non disponible: {e}")
        return {
            "corpus": corpus_name,
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Évaluer modèle QAT/quantifié")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Chemin vers modèle QAT/quantifié",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="bofenghuang/whisper-large-v3-distil-fr-v0.2",
        help="Modèle baseline pour comparaison",
    )
    parser.add_argument(
        "--corpora",
        type=str,
        nargs="+",
        default=["community-v2", "mtedx", "zaion5", "zaion6"],
        help="Corpus à évaluer (comme dans model card)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Fichier JSON de test (alternative aux corpus standards)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluations/qat",
        help="Répertoire de sortie",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["auto", "onnx", "pytorch", "transformers"],
        default="auto",
        help="Type de modèle",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model_path": args.model_path,
        "baseline_model": args.baseline_model,
        "evaluations": {},
    }
    
    # Charger modèle QAT/quantifié
    print(f"Chargement modèle QAT: {args.model_path}")
    model_qat, processor_qat = load_quantized_model(args.model_path, args.model_type)
    
    # Créer évaluateur
    evaluator_qat = WhisperEvaluator(
        model=model_qat,
        processor=processor_qat,
        device=args.device,
    )
    
    # Évaluer sur corpus standards
    for corpus in args.corpora:
        result = evaluate_on_corpus(model_qat, processor_qat, corpus)
        results["evaluations"][corpus] = result
    
    # Évaluer sur test_data si fourni
    if args.test_data:
        print(f"\n📊 Évaluation sur test_data: {args.test_data}")
        
        with open(args.test_data, 'r') as f:
            test_data = json.load(f)
        
        audio_files = [item["audio"] for item in test_data]
        references = [item["text"] for item in test_data]
        
        eval_result = evaluator_qat.evaluate_on_dataset(
            audio_files,
            references,
            return_individual=False,
        )
        
        results["evaluations"]["test_data"] = {
            "metrics": eval_result["metrics"],
            "performance": eval_result["metrics"]["performance"],
        }
    
    # Évaluer baseline pour comparaison
    print(f"\n📊 Évaluation baseline: {args.baseline_model}")
    
    try:
        processor_baseline = WhisperProcessor.from_pretrained(args.baseline_model)
        model_baseline = WhisperForConditionalGeneration.from_pretrained(args.baseline_model)
        
        evaluator_baseline = WhisperEvaluator(
            model=model_baseline,
            processor=processor_baseline,
            device=args.device,
        )
        
        if args.test_data:
            baseline_result = evaluator_baseline.evaluate_on_dataset(
                audio_files,
                references,
                return_individual=False,
            )
            
            results["baseline"] = {
                "metrics": baseline_result["metrics"],
                "performance": baseline_result["metrics"]["performance"],
            }
            
            # Comparaison
            qat_wer = results["evaluations"]["test_data"]["metrics"]["overall_wer"]
            baseline_wer = results["baseline"]["metrics"]["overall_wer"]
            degradation = qat_wer - baseline_wer
            
            results["comparison"] = {
                "baseline_wer": baseline_wer,
                "qat_wer": qat_wer,
                "degradation": degradation,
                "degradation_percent": (degradation / baseline_wer * 100) if baseline_wer > 0 else 0,
            }
            
            print(f"\n{'='*60}")
            print("COMPARAISON QAT vs BASELINE")
            print(f"{'='*60}")
            print(f"Baseline WER: {baseline_wer:.2f}%")
            print(f"QAT WER:      {qat_wer:.2f}%")
            print(f"Dégradation:  {degradation:+.2f}% ({degradation/baseline_wer*100:+.2f}%)")
            print(f"{'='*60}")
            
            if abs(degradation) < 0.5:
                print("✅ Excellent: Dégradation < 0.5%")
            elif abs(degradation) < 1.0:
                print("✅ Bon: Dégradation < 1.0%")
            elif abs(degradation) < 2.0:
                print("⚠️  Acceptable: Dégradation < 2.0%")
            else:
                print("❌ Dégradation élevée, à améliorer")
    
    except Exception as e:
        print(f"⚠️  Impossible d'évaluer baseline: {e}")
    
    # Sauvegarder résultats
    output_file = output_dir / "qat_evaluation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Résultats sauvegardés dans {output_file}")


if __name__ == "__main__":
    main()

