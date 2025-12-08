#!/usr/bin/env python3
"""
Version optimisée de l'entraînement QAT pour réduire le temps.
Utilise moins de données et optimisations pour accélérer.
"""

import argparse
import torch
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Utiliser le même script mais avec paramètres optimisés
from scripts.train_qat import prepare_qat_model, compute_metrics_qat
from src.data.dataset import prepare_dataset
from src.training.trainer import DataCollatorSpeechSeq2SeqWithPadding
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


def estimate_training_time(
    dataset_size_hours: float,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation: int,
    has_gpu: bool = True,
) -> dict:
    """
    Estime le temps d'entraînement.
    
    Args:
        dataset_size_hours: Taille du dataset en heures
        batch_size: Batch size par device
        num_epochs: Nombre d'époques
        gradient_accumulation: Gradient accumulation steps
        has_gpu: GPU disponible
    
    Returns:
        Dict avec estimations de temps
    """
    # Estimations réalistes
    if has_gpu:
        # GPU moderne (A100/V100) : Whisper distillé fait ~0.05-0.1x RTF (très rapide)
        # Pour training (avec backward pass), on est ~3-5x plus lent que inference
        # Donc ~0.3-0.5x RTF pour training
        processing_speed = 0.4  # real-time factor pour training sur GPU
    else:
        # CPU : ~5-10x real-time
        processing_speed = 7.0
    
    # Temps par epoch
    effective_batch_size = batch_size * gradient_accumulation
    samples_per_epoch = dataset_size_hours * 3600 / 30  # Assume 30s segments
    
    # Temps par epoch (en heures)
    time_per_epoch_hours = (dataset_size_hours * processing_speed) / 3600
    
    # Temps total
    total_time_hours = time_per_epoch_hours * num_epochs
    
    return {
        "time_per_epoch_hours": time_per_epoch_hours,
        "total_time_hours": total_time_hours,
        "total_time_days": total_time_hours / 24,
        "processing_speed_rtf": processing_speed,
        "dataset_size_hours": dataset_size_hours,
        "num_epochs": num_epochs,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimer temps QAT")
    parser.add_argument(
        "--dataset_size_hours",
        type=float,
        default=1000.0,
        help="Taille dataset en heures",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Nombre d'époques",
    )
    parser.add_argument(
        "--has_gpu",
        action="store_true",
        help="GPU disponible",
    )
    
    args = parser.parse_args()
    
    # Vérifier GPU
    has_gpu = args.has_gpu or torch.cuda.is_available()
    
    estimate = estimate_training_time(
        dataset_size_hours=args.dataset_size_hours,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gradient_accumulation=8,
        has_gpu=has_gpu,
    )
    
    print(f"\n{'='*60}")
    print("ESTIMATION TEMPS D'ENTRAÎNEMENT QAT")
    print(f"{'='*60}")
    print(f"\nDataset: {estimate['dataset_size_hours']:.1f} heures")
    print(f"Époques: {estimate['num_epochs']}")
    print(f"Device: {'GPU' if has_gpu else 'CPU'}")
    print(f"Vitesse: {estimate['processing_speed_rtf']:.2f}x real-time")
    
    print(f"\n⏱️  Temps estimé:")
    print(f"  Par époque: {estimate['time_per_epoch_hours']:.2f} heures")
    print(f"  Total: {estimate['total_time_hours']:.2f} heures ({estimate['total_time_days']:.2f} jours)")
    
    print(f"\n💡 OPTIMISATIONS POSSIBLES:")
    print(f"  1. Réduire dataset à 500h: {estimate_training_time(args.dataset_size_hours/2, args.batch_size, args.num_epochs, 8, has_gpu)['total_time_hours']:.1f}h")
    print(f"  2. Réduire à 5 époques: {estimate_training_time(args.dataset_size_hours, args.batch_size, 5, 8, has_gpu)['total_time_hours']:.1f}h")
    print(f"  3. Batch size 8: {estimate_training_time(args.dataset_size_hours, 8, args.num_epochs, 4, has_gpu)['total_time_hours']:.1f}h")
    print(f"  4. Combinaison: 500h + 5 epochs = {estimate_training_time(args.dataset_size_hours/2, args.batch_size, 5, 8, has_gpu)['total_time_hours']:.1f}h")


if __name__ == "__main__":
    main()

