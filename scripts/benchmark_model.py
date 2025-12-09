#!/usr/bin/env python3
"""
Benchmark et métriques pour comparer les modèles (original vs quantifié)
"""

import argparse
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pathlib import Path
import numpy as np

def benchmark_model(model, processor, device="cuda", num_runs=5):
    """Benchmark vitesse d'inférence"""
    model.eval()
    model.to(device)
    
    # Créer des features audio dummy (30 secondes à 16kHz)
    dummy_audio = np.random.randn(480000).astype(np.float32)  # 30s * 16000 Hz
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    
    # Convertir au bon dtype (même que le modèle)
    model_dtype = next(model.parameters()).dtype if hasattr(model, 'parameters') and next(model.parameters(), None) is not None else torch.float16
    inputs = {k: v.to(device).to(model_dtype) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point else v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_length=100)
    
    # Benchmark
    times = []
    torch.cuda.synchronize() if device == "cuda" else None
    
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_length=100)
        torch.cuda.synchronize() if device == "cuda" else None
        end = time.time()
        times.append(end - start)
    
    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "throughput": 30.0 / np.mean(times),  # secondes audio par seconde réelle
    }


def get_model_size(model_path_or_name, is_quantized=False):
    """Calculer la taille du modèle"""
    if is_quantized and Path(model_path_or_name).exists():
        # Taille des fichiers ONNX quantifiés
        total_size = sum(
            f.stat().st_size 
            for f in Path(model_path_or_name).rglob("*.onnx") 
            if f.is_file()
        )
    else:
        # Charger le modèle pour calculer taille
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path_or_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            # Estimer taille (FP16 = 2 bytes par paramètre)
            num_params = sum(p.numel() for p in model.parameters())
            total_size = num_params * 2  # FP16
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"  ⚠️  Erreur calcul taille: {e}")
            total_size = 0
    
    return total_size


def main():
    parser = argparse.ArgumentParser(description="Benchmark modèles Whisper")
    parser.add_argument(
        "--model",
        type=str,
        default="bofenghuang/whisper-large-v3-distil-fr-v0.2",
        help="Modèle à benchmarker",
    )
    parser.add_argument(
        "--quantized",
        type=str,
        default=None,
        help="Chemin vers modèle quantifié (optionnel)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Nombre de runs pour benchmark",
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("📊 BENCHMARK ET MÉTRIQUES DES MODÈLES")
    print("="*70)
    print()
    
    # Vérifier GPU
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA non disponible, utilisation CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Mémoire VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print()
    
    # Modèle original
    print("📦 Modèle original:")
    print(f"   {args.model}")
    
    try:
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Taille
        size_gb = get_model_size(args.model, is_quantized=False) / 1e9
        print(f"   Taille: {size_gb:.2f} GB (FP16)")
        
        # Benchmark
        print("   Benchmark vitesse...")
        metrics_original = benchmark_model(model, processor, args.device, args.num_runs)
        print(f"   Temps moyen: {metrics_original['mean_time']:.2f}s ± {metrics_original['std_time']:.2f}s")
        print(f"   Débit: {metrics_original['throughput']:.2f}x temps réel")
        
        # Mémoire GPU
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            test_inputs = processor(np.random.randn(48000), sampling_rate=16000, return_tensors="pt")
            test_inputs = {k: v.to(args.device).to(torch.float16) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point else v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in test_inputs.items()}
            _ = model.generate(**test_inputs, max_length=50)
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"   Mémoire VRAM max: {peak_memory:.2f} GB")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print()
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        metrics_original = None
        size_gb = 0
    
    # Modèle quantifié (si fourni)
    if args.quantized and Path(args.quantized).exists():
        print("📦 Modèle quantifié:")
        print(f"   {args.quantized}")
        
        try:
            from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
            
            # Détecter automatiquement les noms de fichiers ONNX
            quantized_path = Path(args.quantized)
            encoder_files = list(quantized_path.glob("encoder_model*.onnx"))
            decoder_files = list(quantized_path.glob("decoder_model*.onnx"))
            
            encoder_name = "encoder_model_quantized.onnx" if any("quantized" in f.name for f in encoder_files) else "encoder_model.onnx"
            decoder_name = "decoder_model_quantized.onnx" if any("quantized" in f.name for f in decoder_files) else "decoder_model.onnx"
            
            # Vérifier que les fichiers existent
            if not (quantized_path / encoder_name).exists():
                encoder_name = encoder_files[0].name if encoder_files else "encoder_model.onnx"
            if not (quantized_path / decoder_name).exists():
                decoder_name = decoder_files[0].name if decoder_files else "decoder_model.onnx"
            
            model_quantized = ORTModelForSpeechSeq2Seq.from_pretrained(
                args.quantized,
                encoder_file_name=encoder_name,
                decoder_file_name=decoder_name,
                use_cache=False,
            )
            processor_quantized = AutoProcessor.from_pretrained(args.quantized)
            
            # Taille
            size_quantized_gb = get_model_size(args.quantized, is_quantized=True) / 1e9
            print(f"   Taille: {size_quantized_gb:.2f} GB (int8)")
            
            if metrics_original:
                reduction = (1 - size_quantized_gb / size_gb) * 100
                print(f"   Réduction: {reduction:.1f}% vs original")
            
            # Benchmark
            print("   Benchmark vitesse...")
            metrics_quantized = benchmark_model(model_quantized, processor_quantized, args.device, args.num_runs)
            print(f"   Temps moyen: {metrics_quantized['mean_time']:.2f}s ± {metrics_quantized['std_time']:.2f}s")
            print(f"   Débit: {metrics_quantized['throughput']:.2f}x temps réel")
            
            if metrics_original:
                speedup = metrics_original['mean_time'] / metrics_quantized['mean_time']
                print(f"   Accélération: {speedup:.2f}x vs original")
            
            print()
            
            # Résumé comparatif
            if metrics_original:
                print("="*70)
                print("📈 RÉSUMÉ COMPARATIF")
                print("="*70)
                print()
                print(f"Taille:")
                print(f"  Original (FP16):  {size_gb:.2f} GB")
                print(f"  Quantifié (int8): {size_quantized_gb:.2f} GB")
                print(f"  Réduction:        {(1 - size_quantized_gb / size_gb) * 100:.1f}%")
                print()
                print(f"Vitesse (30s audio):")
                print(f"  Original:         {metrics_original['mean_time']:.2f}s")
                print(f"  Quantifié:        {metrics_quantized['mean_time']:.2f}s")
                print(f"  Accélération:     {metrics_original['mean_time'] / metrics_quantized['mean_time']:.2f}x")
                print()
                
        except Exception as e:
            print(f"   ❌ Erreur chargement quantifié: {e}")
            print("   (Le modèle quantifié n'est peut-être pas compatible)")
    
    else:
        print()
        print("💡 Pour comparer avec modèle quantifié:")
        print(f"   python scripts/benchmark_model.py --quantized outputs/models/whisper-ptq-int8/quantized")
    
    print("="*70)


if __name__ == "__main__":
    main()

