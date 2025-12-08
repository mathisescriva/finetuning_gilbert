#!/usr/bin/env python3
"""
Benchmark simplifié - vitesse et mémoire uniquement (pas besoin de datasets)
"""

import argparse
import time
import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from tqdm import tqdm


def benchmark_model(model, processor, device="cuda", num_runs=10, audio_length_seconds=30):
    """Benchmark vitesse avec audio synthétique"""
    model.eval()
    model.to(device)
    
    # Créer audio synthétique
    sampling_rate = 16000
    audio_length = audio_length_seconds * sampling_rate
    dummy_audio = np.random.randn(audio_length).astype(np.float32)
    
    times = []
    memory_usage = []
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1e9
    
    # Warmup
    inputs = processor(dummy_audio, sampling_rate=sampling_rate, return_tensors="pt")
    model_dtype = next(model.parameters()).dtype
    inputs_processed = {
        k: v.to(device).to(model_dtype) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point 
        else v.to(device) if isinstance(v, torch.Tensor) else v 
        for k, v in inputs.items()
    }
    
    with torch.no_grad():
        _ = model.generate(**inputs_processed, max_length=100)
    
    # Benchmark
    for _ in tqdm(range(num_runs), desc="Benchmark"):
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.time()
        
        with torch.no_grad():
            _ = model.generate(**inputs_processed, max_length=448, language="fr")
        
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
        "p50_time": np.median(times),
        "p95_time": np.percentile(times, 95),
        "p99_time": np.percentile(times, 99),
        "throughput_realtime": audio_length_seconds / np.mean(times),
        "peak_memory_gb": memory_usage[0] if memory_usage else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark simplifié - vitesse uniquement")
    parser.add_argument(
        "--model",
        type=str,
        default="bofenghuang/whisper-large-v3-distil-fr-v0.2",
        help="Modèle à benchmarker",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluations/benchmark_results.json",
        help="Fichier de sortie JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=20,
        help="Nombre de runs pour statistiques robustes",
    )
    parser.add_argument(
        "--audio_length",
        type=int,
        default=30,
        help="Longueur audio en secondes",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("⚡ BENCHMARK VITESSE ET MÉMOIRE")
    print("="*80)
    print()
    
    # Vérifier GPU
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA non disponible, utilisation CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Mémoire VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print()
    
    # Charger modèle
    print(f"📦 Chargement modèle: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    # Métriques de taille
    num_params = sum(p.numel() for p in model.parameters())
    model_size_gb = num_params * 2 / 1e9  # FP16
    
    print(f"✅ Modèle chargé")
    print(f"   Paramètres: {num_params / 1e6:.1f}M")
    print(f"   Taille: {model_size_gb:.2f} GB (FP16)")
    print()
    
    # Benchmark
    print(f"⚡ Benchmark vitesse ({args.num_runs} runs, {args.audio_length}s audio)...")
    bench_results = benchmark_model(model, processor, args.device, args.num_runs, args.audio_length)
    
    print()
    print("📊 Résultats:")
    print(f"   Temps moyen: {bench_results['mean_time']:.3f}s ± {bench_results['std_time']:.3f}s")
    print(f"   Temps médian: {bench_results['p50_time']:.3f}s")
    print(f"   Temps min: {bench_results['min_time']:.3f}s")
    print(f"   Temps max: {bench_results['max_time']:.3f}s")
    print(f"   P95: {bench_results['p95_time']:.3f}s")
    print(f"   Débit: {bench_results['throughput_realtime']:.1f}x temps réel")
    if bench_results['peak_memory_gb']:
        print(f"   Mémoire VRAM: {bench_results['peak_memory_gb']:.2f} GB")
    print()
    
    # Préparer résultats
    results = {
        "model_name": args.model,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
        "num_parameters": int(num_params),
        "model_size_gb": model_size_gb,
        "audio_length_seconds": args.audio_length,
        "num_runs": args.num_runs,
        "inference_benchmark": bench_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Sauvegarder
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Résultats sauvegardés: {output_path}")
    print()
    
    # Résumé pour papier
    print("="*80)
    print("📄 RÉSUMÉ POUR PUBLICATION")
    print("="*80)
    print()
    print(f"Modèle: {args.model}")
    print(f"Taille: {model_size_gb:.2f} GB ({num_params/1e6:.1f}M paramètres)")
    print(f"Vitesse: {bench_results['mean_time']:.3f}s pour {args.audio_length}s audio")
    print(f"Débit: {bench_results['throughput_realtime']:.1f}x temps réel")
    if bench_results['peak_memory_gb']:
        print(f"Mémoire VRAM: {bench_results['peak_memory_gb']:.2f} GB")
    print()
    print("💡 Pour métriques qualité (WER/CER), utilisez:")
    print("   python scripts/evaluate_comprehensive.py (nécessite datasets avec transcripts)")
    print()
    print("="*80)


if __name__ == "__main__":
    main()

