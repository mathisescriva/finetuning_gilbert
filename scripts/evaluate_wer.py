#!/usr/bin/env python3
"""
Script pour mesurer WER/CER sur un dataset avec transcripts
"""

import argparse
import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset, Dataset
from jiwer import wer, cer
from tqdm import tqdm
import os

# Configurer temp directory
os.environ["TMPDIR"] = os.path.expanduser("~/.cache")
os.environ["TEMP"] = os.path.expanduser("~/.cache")
os.environ["TMP"] = os.path.expanduser("~/.cache")


def evaluate_wer(model, processor, dataset, device="cuda", max_samples=None):
    """Mesure WER et CER sur un dataset"""
    model.eval()
    model.to(device)
    
    predictions = []
    references = []
    
    samples = dataset.select(range(min(max_samples, len(dataset)))) if max_samples else dataset
    
    print(f"📊 Évaluation sur {len(samples)} échantillons...")
    
    for sample in tqdm(samples, desc="Transcription"):
        # Audio
        audio = sample.get("audio", {})
        if isinstance(audio, dict):
            audio_array = audio.get("array", None)
            sr = audio.get("sampling_rate", 16000)
        else:
            audio_array = audio
            sr = 16000
        
        if audio_array is None:
            continue
        
        # Référence
        reference = sample.get("text", sample.get("sentence", ""))
        if not reference:
            continue
        
        # Convertir en numpy si nécessaire
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        
        # Transcription
        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")
        
        # Convertir au bon dtype
        model_dtype = next(model.parameters()).dtype
        inputs_processed = {
            k: v.to(device).to(model_dtype) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point 
            else v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in inputs.items()
        }
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs_processed,
                max_length=448,
                language="fr",
            )
        
        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        predictions.append(prediction)
        references.append(reference)
    
    # Calculer métriques
    wer_score = wer(references, predictions)
    cer_score = cer(references, predictions)
    
    return {
        "wer": wer_score,
        "cer": cer_score,
        "num_samples": len(predictions),
        "predictions": predictions[:5],  # Exemples
        "references": references[:5],
    }


def main():
    parser = argparse.ArgumentParser(description="Mesurer WER/CER")
    parser.add_argument(
        "--model",
        type=str,
        default="bofenghuang/whisper-large-v3-distil-fr-v0.2",
        help="Modèle à évaluer",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset HuggingFace (ex: mozilla-foundation/common_voice_13_0) ou chemin local",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="fr",
        help="Config du dataset (ex: 'fr' pour Common Voice)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split du dataset (test/validation)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Nombre max d'échantillons",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluations/wer_results.json",
        help="Fichier de sortie",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("📊 MESURE WER/CER")
    print("="*80)
    print()
    
    # Charger modèle
    print(f"📦 Chargement modèle: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print("✅ Modèle chargé")
    print()
    
    # Charger dataset
    print(f"📥 Chargement dataset: {args.dataset}")
    try:
        if Path(args.dataset).exists():
            # Dataset local
            dataset = load_dataset("json", data_files=args.dataset, split="train")
        else:
            # Dataset HuggingFace
            dataset = load_dataset(
                args.dataset,
                args.dataset_config,
                split=args.split,
                streaming=False,
            )
        print(f"✅ Dataset chargé: {len(dataset)} échantillons")
    except Exception as e:
        print(f"❌ Erreur chargement dataset: {e}")
        return
    
    # Évaluer
    results = evaluate_wer(
        model, processor, dataset, args.device, max_samples=args.max_samples
    )
    
    print()
    print("="*80)
    print("📊 RÉSULTATS WER/CER")
    print("="*80)
    print()
    print(f"Dataset: {args.dataset}")
    print(f"Échantillons évalués: {results['num_samples']}")
    print()
    print(f"WER: {results['wer']:.2%}")
    print(f"CER: {results['cer']:.2%}")
    print()
    
    # Sauvegarder
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Résultats sauvegardés: {output_path}")


if __name__ == "__main__":
    main()

