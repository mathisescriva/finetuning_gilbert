#!/usr/bin/env python3
"""
Évaluation complète pour publication scientifique
Mesure: WER, CER, latence, mémoire, comparaisons
"""

import argparse
import time
import os
import torch
import json
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset, Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
from jiwer import wer, cer
import sys

# Configurer répertoire temporaire - utiliser /workspace si disponible, sinon /tmp
# /workspace est souvent sur un disque avec plus d'espace
temp_dirs = ["/workspace/tmp", "/tmp"]
temp_dir = None
for td in temp_dirs:
    try:
        os.makedirs(td, exist_ok=True)
        # Test write
        test_file = os.path.join(td, ".test_write")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        temp_dir = td
        break
    except (OSError, PermissionError):
        continue

if temp_dir:
    os.environ["TMPDIR"] = temp_dir
    os.environ["TEMP"] = temp_dir
    os.environ["TMP"] = temp_dir
else:
    # Fallback: utiliser le répertoire courant
    os.environ["TMPDIR"] = str(Path.cwd() / "tmp")
    os.environ["TEMP"] = str(Path.cwd() / "tmp")
    os.environ["TMP"] = str(Path.cwd() / "tmp")
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

sys.path.append(str(Path(__file__).parent.parent / "src"))
# Note: on utilise jiwer directement pour WER/CER


def benchmark_inference(model, processor, audio_samples, device="cuda", batch_size=1):
    """Benchmark vitesse et mémoire"""
    model.eval()
    model.to(device)
    
    times = []
    memory_usage = []
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1e9
    
        # Traiter par batch
    for i in tqdm(range(0, len(audio_samples), batch_size), desc="Benchmark vitesse"):
        batch = audio_samples[i:i+batch_size]
        
        # Préparer inputs
        inputs_list = []
        for audio in batch:
            # Extraire audio array et sampling rate
            if isinstance(audio, dict):
                if "audio" in audio and isinstance(audio["audio"], dict):
                    audio_array = audio["audio"].get("array", None)
                    sr = audio["audio"].get("sampling_rate", 16000)
                else:
                    audio_array = audio.get("array", audio.get("audio"))
                    sr = audio.get("sampling_rate", 16000)
            else:
                audio_array = audio
                sr = 16000
            
            # S'assurer qu'on a un array numpy
            if audio_array is None:
                continue
            
            # Convertir en numpy array si nécessaire
            if not isinstance(audio_array, np.ndarray):
                import numpy as np
                audio_array = np.array(audio_array, dtype=np.float32)
            
            inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")
            inputs_list.append(inputs)
        
        # Traiter chaque input (Whisper ne supporte pas vraiment le batching)
        for inputs in inputs_list:
            # Convertir au bon dtype
            model_dtype = next(model.parameters()).dtype if hasattr(model, 'parameters') else torch.float16
            inputs_processed = {
                k: v.to(device).to(model_dtype) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point 
                else v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()
            }
            
            # Mesurer temps
            torch.cuda.synchronize() if device == "cuda" else None
            start = time.time()
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs_processed,
                    max_length=448,
                    language="fr",
                )
            
            torch.cuda.synchronize() if device == "cuda" else None
            end = time.time()
            
            times.append(end - start)
    
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        memory_usage.append(peak_memory - initial_memory)
    
    return {
        "times": times,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "peak_memory_gb": memory_usage[0] if memory_usage else None,
    }


def evaluate_quality(model, processor, dataset, device="cuda", max_samples=None):
    """Évaluer qualité (WER, CER)"""
    model.eval()
    model.to(device)
    
    predictions = []
    references = []
    
    samples = dataset.select(range(min(max_samples, len(dataset)))) if max_samples else dataset
    
    for sample in tqdm(samples, desc="Évaluation qualité"):
        # Audio - gérer différents formats
        if isinstance(sample, dict):
            audio = sample.get("audio", {})
            if isinstance(audio, dict):
                audio_array = audio.get("array", None)
                sr = audio.get("sampling_rate", 16000)
            else:
                audio_array = audio
                sr = 16000
        else:
            audio_array = sample
            sr = 16000
        
        # Référence
        reference = sample.get("text", sample.get("sentence", ""))
        if not reference:
            continue
        
        # S'assurer qu'on a un array numpy
        if audio_array is None:
            continue
        
        # Convertir en numpy array si nécessaire
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        
        # Transcription
        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")
        
        # Convertir au bon dtype
        model_dtype = next(model.parameters()).dtype if hasattr(model, 'parameters') else torch.float16
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
        "predictions": predictions[:10],  # Garder quelques exemples
        "references": references[:10],
    }


def load_test_datasets(max_samples_per_dataset=100):
    """Charger datasets de test"""
    datasets = {}
    
    # Common Voice French (si disponible)
    try:
        print("📥 Chargement Common Voice French...")
        cv_fr = load_dataset("mozilla-foundation/common_voice_13_0", "fr", split="test", trust_remote_code=True)
        if len(cv_fr) > 0:
            datasets["common_voice_fr"] = cv_fr.select(range(min(max_samples_per_dataset, len(cv_fr))))
            print(f"   ✅ {len(datasets['common_voice_fr'])} échantillons")
    except Exception as e:
        print(f"   ⚠️  Common Voice non disponible: {e}")
    
    # MLS French
    try:
        print("📥 Chargement MLS French...")
        mls_fr = load_dataset("facebook/multilingual_librispeech", "french", split="test", trust_remote_code=True)
        if len(mls_fr) > 0:
            datasets["mls_fr"] = mls_fr.select(range(min(max_samples_per_dataset, len(mls_fr))))
            print(f"   ✅ {len(datasets['mls_fr'])} échantillons")
    except Exception as e:
        print(f"   ⚠️  MLS non disponible: {e}")
    
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Évaluation complète pour publication")
    parser.add_argument(
        "--model",
        type=str,
        default="bofenghuang/whisper-large-v3-distil-fr-v0.2",
        help="Modèle à évaluer",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="openai/whisper-large-v3",
        help="Modèle baseline pour comparaison",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluations/comprehensive_results.json",
        help="Fichier de sortie JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Nombre max d'échantillons par dataset",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline model (plus rapide)",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("📊 ÉVALUATION COMPLÈTE POUR PUBLICATION")
    print("="*80)
    print()
    
    # Créer répertoire de sortie
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model_name": args.model,
        "baseline_model": args.baseline_model,
        "device": args.device,
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Charger datasets de test
    print("📦 Chargement des datasets de test...")
    test_datasets = load_test_datasets(args.max_samples)
    print()
    
    if not test_datasets:
        print("❌ Aucun dataset de test disponible")
        print("💡 Création d'un dataset minimal pour test...")
        # Créer dataset minimal pour benchmark vitesse seulement
    # Note: Pour qualité (WER/CER), il faut un vrai dataset avec transcripts
    dummy_audio = np.random.randn(16000).astype(np.float32)
    dummy_samples = [{
        "audio": {"array": dummy_audio, "sampling_rate": 16000},
        "text": "Test de transcription"
    }]
    test_datasets = {"dummy_benchmark": dummy_samples}
    
    # Évaluer modèle principal
    print(f"🔍 Évaluation du modèle: {args.model}")
    print("-" * 80)
    
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    # Métriques de taille
    num_params = sum(p.numel() for p in model.parameters())
    model_size_gb = num_params * 2 / 1e9  # FP16 = 2 bytes
    results["model_size_gb"] = model_size_gb
    results["num_parameters"] = num_params
    
    print(f"📊 Taille du modèle:")
    print(f"   Paramètres: {num_params / 1e6:.1f}M")
    print(f"   Taille: {model_size_gb:.2f} GB (FP16)")
    print()
    
    # Benchmark vitesse
    print("⚡ Benchmark vitesse...")
    audio_samples = []
    for dataset_name, dataset in test_datasets.items():
        for sample in dataset.select(range(min(10, len(dataset)))):
            audio_samples.append(sample)
    
    if audio_samples:
        bench_results = benchmark_inference(model, processor, audio_samples, args.device)
        results["inference_benchmark"] = bench_results
        
        print(f"   Temps moyen: {bench_results['mean_time']:.3f}s ± {bench_results['std_time']:.3f}s")
        print(f"   Mémoire VRAM: {bench_results['peak_memory_gb']:.2f} GB" if bench_results['peak_memory_gb'] else "")
        print()
    
    # Évaluation qualité par dataset
    print("🎯 Évaluation qualité (WER/CER)...")
    quality_results = {}
    
    for dataset_name, dataset in test_datasets.items():
        print(f"\n   Dataset: {dataset_name}")
        try:
            eval_results = evaluate_quality(
                model, processor, dataset, args.device, max_samples=args.max_samples
            )
            quality_results[dataset_name] = eval_results
            
            print(f"      WER: {eval_results['wer']:.2%}")
            print(f"      CER: {eval_results['cer']:.2%}")
            print(f"      Échantillons: {eval_results['num_samples']}")
        except Exception as e:
            print(f"      ❌ Erreur: {e}")
            quality_results[dataset_name] = {"error": str(e)}
    
    results["quality_metrics"] = quality_results
    
    # Calculer moyenne pondérée
    total_samples = sum(r.get("num_samples", 0) for r in quality_results.values())
    if total_samples > 0:
        avg_wer = sum(r.get("wer", 0) * r.get("num_samples", 0) for r in quality_results.values()) / total_samples
        avg_cer = sum(r.get("cer", 0) * r.get("num_samples", 0) for r in quality_results.values()) / total_samples
        results["average_wer"] = avg_wer
        results["average_cer"] = avg_cer
        
        print()
        print(f"📊 Moyenne pondérée:")
        print(f"   WER moyen: {avg_wer:.2%}")
        print(f"   CER moyen: {avg_cer:.2%}")
    
    # Comparaison avec baseline (si demandé)
    if not args.skip_baseline:
        print()
        print("-" * 80)
        print(f"🔍 Évaluation baseline: {args.baseline_model}")
        
        try:
            baseline_processor = AutoProcessor.from_pretrained(args.baseline_model)
            baseline_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                args.baseline_model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            
            # Taille baseline
            baseline_num_params = sum(p.numel() for p in baseline_model.parameters())
            baseline_size_gb = baseline_num_params * 2 / 1e9
            results["baseline_size_gb"] = baseline_size_gb
            results["baseline_num_parameters"] = baseline_num_params
            
            print(f"   Taille: {baseline_size_gb:.2f} GB")
            
            # Vitesse baseline (échantillon réduit)
            if audio_samples:
                baseline_bench = benchmark_inference(
                    baseline_model, baseline_processor, audio_samples[:5], args.device
                )
                results["baseline_inference"] = baseline_bench
                
                speedup = baseline_bench["mean_time"] / bench_results["mean_time"]
                size_reduction = (1 - model_size_gb / baseline_size_gb) * 100
                
                results["speedup_vs_baseline"] = speedup
                results["size_reduction_percent"] = size_reduction
                
                print(f"   Vitesse baseline: {baseline_bench['mean_time']:.3f}s")
                print(f"   Accélération: {speedup:.2f}x")
                print(f"   Réduction taille: {size_reduction:.1f}%")
            
            # Qualité baseline (sur un seul dataset pour gagner du temps)
            if test_datasets:
                first_dataset = list(test_datasets.values())[0]
                baseline_quality = evaluate_quality(
                    baseline_model, baseline_processor, first_dataset, args.device, max_samples=min(50, args.max_samples)
                )
                results["baseline_quality"] = baseline_quality
                
                if dataset_name in quality_results:
                    wer_degradation = quality_results[dataset_name]["wer"] - baseline_quality["wer"]
                    results["wer_degradation_vs_baseline"] = wer_degradation
                    
                    print(f"   WER baseline: {baseline_quality['wer']:.2%}")
                    print(f"   Dégradation WER: {wer_degradation:+.2%}")
            
            del baseline_model
            torch.cuda.empty_cache() if args.device == "cuda" else None
            
        except Exception as e:
            print(f"   ❌ Erreur évaluation baseline: {e}")
            results["baseline_error"] = str(e)
    
    # Sauvegarder résultats
    print()
    print("="*80)
    print("💾 Sauvegarde des résultats...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Résultats sauvegardés dans: {output_path}")
    print()
    
    # Résumé pour papier
    print("="*80)
    print("📄 RÉSUMÉ POUR PUBLICATION")
    print("="*80)
    print()
    print(f"Modèle: {args.model}")
    print(f"Taille: {model_size_gb:.2f} GB ({num_params/1e6:.1f}M paramètres)")
    if "speedup_vs_baseline" in results:
        print(f"Accélération vs large-v3: {results['speedup_vs_baseline']:.2f}x")
        print(f"Réduction taille: {results['size_reduction_percent']:.1f}%")
    if "average_wer" in results:
        print(f"WER moyen: {results['average_wer']:.2%}")
        print(f"CER moyen: {results['average_cer']:.2%}")
    if "wer_degradation_vs_baseline" in results:
        print(f"Dégradation WER vs baseline: {results['wer_degradation_vs_baseline']:+.2%}")
    print()
    print("="*80)


if __name__ == "__main__":
    main()

