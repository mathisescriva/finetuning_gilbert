#!/usr/bin/env python3
"""
Script de benchmark complet comparant différents modèles.
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.evaluator import WhisperEvaluator


def benchmark_model(
    model_name: str,
    audio_files: list,
    references: list,
    entity_list: set = None,
    device: str = "cuda",
    config_path: str = "config/model_config.yaml",
):
    """
    Benchmark un modèle complet.
    
    Returns:
        Dict avec toutes les métriques
    """
    import yaml
    
    # Charger config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Charger modèle
    print(f"Chargement {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Créer évaluateur
    inference_config = config.get("inference", {})
    evaluator = WhisperEvaluator(
        model=model,
        processor=processor,
        device=device,
        chunk_length_s=inference_config.get("chunk_length_s", 30),
        beam_size=inference_config.get("beam_size", 5),
    )
    
    # Évaluation
    results = evaluator.evaluate_on_dataset(
        audio_files,
        references,
        entity_list=entity_list,
    )
    
    # Mémoire
    memory = evaluator.get_model_memory_usage()
    
    # Benchmark détaillé
    if len(audio_files) > 0:
        benchmark = evaluator.benchmark_inference(audio_files[0])
    else:
        benchmark = {}
    
    return {
        "model_name": model_name,
        "metrics": results["metrics"],
        "performance": results["metrics"]["performance"],
        "memory": memory,
        "benchmark": benchmark,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark comparatif")
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Fichier JSON de test",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "openai/whisper-large-v3",
            "bofenghuang/whisper-large-v3-distil-fr-v0.2",
        ],
        help="Liste de modèles à comparer",
    )
    parser.add_argument(
        "--custom_models",
        type=str,
        nargs="+",
        default=[],
        help="Chemins vers modèles locaux custom",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluations",
        help="Répertoire de sortie",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    
    args = parser.parse_args()
    
    # Charger données
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    audio_files = [item["audio"] for item in test_data]
    references = [item["text"] for item in test_data]
    entity_list = set(test_data.get("entities", []))
    
    # Benchmark chaque modèle
    all_results = []
    
    for model_name in args.models:
        try:
            result = benchmark_model(
                model_name,
                audio_files,
                references,
                entity_list,
                args.device,
            )
            all_results.append(result)
        except Exception as e:
            print(f"Erreur avec {model_name}: {e}")
            continue
    
    # Modèles locaux
    for model_path in args.custom_models:
        try:
            result = benchmark_model(
                model_path,
                audio_files,
                references,
                entity_list,
                args.device,
            )
            all_results.append(result)
        except Exception as e:
            print(f"Erreur avec {model_path}: {e}")
            continue
    
    # Sauvegarder résultats
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "benchmark_comparison.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Afficher comparaison
    print("\n" + "="*80)
    print("COMPARAISON DES MODÈLES")
    print("="*80)
    
    print(f"\n{'Modèle':<50} {'WER':<10} {'CER':<10} {'RTF':<10} {'VRAM (GB)':<12}")
    print("-"*80)
    
    for result in all_results:
        model_name = result["model_name"]
        wer = result["metrics"].get("overall_wer", 0)
        cer = result["metrics"].get("overall_cer", 0)
        rtf = result["performance"].get("avg_real_time_factor", 0)
        vram = result["memory"].get("gpu_memory_allocated_gb", 0)
        
        print(f"{model_name:<50} {wer:<10.2f} {cer:<10.2f} {rtf:<10.3f} {vram:<12.2f}")
    
    print("="*80)
    print(f"\nRésultats détaillés sauvegardés dans {output_file}")


if __name__ == "__main__":
    main()

