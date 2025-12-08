#!/usr/bin/env python3
"""
Script d'évaluation du modèle de base (baseline).
Évalue bofenghuang/whisper-large-v3-distil-fr-v0.2 sur données de réunions.
"""

import argparse
import yaml
import json
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.evaluator import WhisperEvaluator


def load_test_data(test_data_path: str):
    """
    Charge les données de test.
    Format attendu: JSON avec liste de {"audio": "path", "text": "transcript"}
    """
    with open(test_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    audio_files = [item["audio"] for item in data]
    references = [item["text"] for item in data]
    
    # Optionnel: liste d'entités
    entity_list = None
    if "entities" in data and isinstance(data["entities"], list):
        entity_list = set(data["entities"])
    
    return audio_files, references, entity_list


def main():
    parser = argparse.ArgumentParser(description="Évaluer le modèle baseline")
    parser.add_argument(
        "--model_name",
        type=str,
        default="bofenghuang/whisper-large-v3-distil-fr-v0.2",
        help="Nom du modèle HuggingFace",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Chemin vers fichier JSON de test",
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
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Chemin vers config modèle",
    )
    
    args = parser.parse_args()
    
    # Créer répertoire de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Charger modèle
    print(f"Chargement du modèle {args.model_name}...")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Créer évaluateur
    inference_config = config.get("inference", {})
    evaluator = WhisperEvaluator(
        model=model,
        processor=processor,
        device=args.device,
        chunk_length_s=inference_config.get("chunk_length_s", 30),
        beam_size=inference_config.get("beam_size", 5),
    )
    
    # Charger données de test
    print(f"Chargement des données de test depuis {args.test_data}...")
    audio_files, references, entity_list = load_test_data(args.test_data)
    
    print(f"Évaluation sur {len(audio_files)} fichiers...")
    
    # Évaluation complète
    results = evaluator.evaluate_on_dataset(
        audio_files=audio_files,
        references=references,
        entity_list=entity_list,
        return_individual=True,
    )
    
    # Mesure mémoire
    memory_usage = evaluator.get_model_memory_usage()
    
    # Benchmark sur un fichier
    if len(audio_files) > 0:
        benchmark = evaluator.benchmark_inference(audio_files[0])
    else:
        benchmark = {}
    
    # Compiler résultats
    full_results = {
        "model_name": args.model_name,
        "evaluation": results["metrics"],
        "performance": results["metrics"]["performance"],
        "memory_usage": memory_usage,
        "benchmark": benchmark,
        "num_samples": len(audio_files),
    }
    
    # Sauvegarder résultats
    output_file = output_dir / "baseline_evaluation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("RÉSULTATS BASELINE")
    print(f"{'='*60}")
    print(f"WER Global: {full_results['evaluation']['overall_wer']:.2f}%")
    print(f"CER Global: {full_results['evaluation']['overall_cer']:.2f}%")
    print(f"WER Entités: {full_results['evaluation']['named_entities_wer']:.2f}%")
    print(f"WER Acronymes: {full_results['evaluation']['acronyms_wer']:.2f}%")
    print(f"\nPerformance:")
    print(f"  Real-time factor: {full_results['performance']['avg_real_time_factor']:.3f}x")
    print(f"  Latence par minute: {full_results['performance']['latency_per_minute']:.2f}s/min")
    print(f"\nMémoire:")
    for k, v in memory_usage.items():
        print(f"  {k}: {v:.2f}")
    print(f"\n{'='*60}")
    print(f"Résultats sauvegardés dans {output_file}")
    
    return full_results


if __name__ == "__main__":
    main()

