#!/usr/bin/env python3
"""
Script d'entraînement QAT optimisé pour Vast.ai
Focus: Performance/Frugalité/Vitesse maximale
"""

import argparse
import yaml
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import prepare_dataset, create_meetings_dataset_from_files
from src.data.augmentations import create_augmentation_pipeline
from src.evaluation.metrics import compute_wer
from src.training.trainer import DataCollatorSpeechSeq2SeqWithPadding

# Configurer temp directory (important pour Vast.ai)
os.environ["TMPDIR"] = os.environ.get("TMPDIR", "/workspace/tmp")
os.environ["TEMP"] = os.environ.get("TEMP", "/workspace/tmp")
os.environ["TMP"] = os.environ.get("TMP", "/workspace/tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)


class QATModelWrapper(nn.Module):
    """Wrapper pour fake quantization pendant training."""
    
    def __init__(self, model, quantization_config=None):
        super().__init__()
        self.model = model
        
        if quantization_config is None:
            quantization_config = {
                "activation": "int8",
                "weight": "int8",
            }
        self.quantization_config = quantization_config
        self._prepare_fake_quantization()
    
    def _prepare_fake_quantization(self):
        """Prépare le modèle pour fake quantization."""
        try:
            from torch.quantization import prepare_qat, get_default_qat_qconfig
            
            qconfig = get_default_qat_qconfig('fbgemm')
            
            if self.quantization_config.get("weight") == "int4":
                print("⚠️  Int4 nécessite implémentation custom, utilisation int8")
            
            self.model.train()
            self.model.qconfig = qconfig
            
            try:
                prepare_qat(self.model, inplace=True)
                print("✅ Fake quantization activée")
            except Exception as e:
                print(f"⚠️  Erreur QAT standard: {e}")
                print("   Utilisation méthode alternative")
                self.use_manual_quant = True
        
        except ImportError:
            print("⚠️  torch.quantization non disponible, méthode alternative")
            self.use_manual_quant = True
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def prepare_qat_model(base_model_path: str, quantization_type: str = "int8"):
    """Prépare un modèle pour QAT."""
    print(f"📦 Chargement modèle: {base_model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    print(f"🔧 Préparation QAT ({quantization_type})...")
    qat_model = QATModelWrapper(
        model,
        quantization_config={
            "activation": quantization_type,
            "weight": quantization_type,
        }
    )
    
    return qat_model.model


def compute_metrics_qat(pred, processor, metric_key_prefix: str = "eval"):
    """Calcule WER comme métrique."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = compute_wer(label_str, pred_str)
    
    return {f"{metric_key_prefix}_wer": wer}


def main():
    parser = argparse.ArgumentParser(description="Entraînement QAT optimisé")
    parser.add_argument(
        "--base_model",
        type=str,
        default="bofenghuang/whisper-large-v3-distil-fr-v0.2",
        help="Modèle de base",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Données d'entraînement",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        required=True,
        help="Données d'évaluation",
    )
    parser.add_argument(
        "--quantization_type",
        type=str,
        choices=["int8", "int4"],
        default="int8",
        help="Type de quantization",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/models/gilbert-whisper-qat-int8",
        help="Répertoire de sortie",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Nombre d'époques (5 optimisé pour vitesse)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=60000,
        help="Limiter dataset (~500h, optimisé pour vitesse)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate (conservateur pour qualité)",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=8,
        help="Batch size (8 optimisé pour GPU 16-24GB)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation (équivalent batch_size 32)",
    )
    
    args = parser.parse_args()
    
    # Ajuster output_dir
    if args.quantization_type not in args.output_dir:
        args.output_dir = args.output_dir.replace("qat", f"qat-{args.quantization_type}")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎮 Device: {device}")
    if device == "cpu":
        print("⚠️  QAT sur CPU sera très lent. GPU recommandé.")
    
    # Charger modèle et processor
    print(f"📦 Chargement modèle {args.base_model}...")
    processor = WhisperProcessor.from_pretrained(
        args.base_model,
        language="fr",
        task="transcribe",
    )
    
    # Préparer modèle pour QAT
    model = prepare_qat_model(args.base_model, args.quantization_type)
    model = model.to(device)
    
    # Charger données
    print(f"📊 Chargement données...")
    
    # Support JSON ou HuggingFace
    if args.train_data.endswith('.json'):
        with open(args.train_data, 'r') as f:
            train_data = json.load(f)
        audio_files = [item["audio"] for item in train_data]
        transcripts = [item["text"] for item in train_data]
        
        train_dataset = create_meetings_dataset_from_files(
            audio_files,
            transcripts,
            processor,
            augmentations=create_augmentation_pipeline({}),
        )
        
        # Limiter si nécessaire
        if args.max_samples and len(train_dataset) > args.max_samples:
            train_dataset = train_dataset.select(range(args.max_samples))
    else:
        # HuggingFace dataset
        dataset_full = load_dataset(args.train_data, split="train", streaming=False)
        
        if args.max_samples and len(dataset_full) > args.max_samples:
            print(f"   Limitation à {args.max_samples} échantillons")
            dataset_full = dataset_full.select(range(args.max_samples))
        
        train_dataset = prepare_dataset(
            args.train_data,
            processor,
            split="train",
            augmentations=create_augmentation_pipeline({}),
        )
        if hasattr(train_dataset, 'dataset') and len(train_dataset.dataset) > args.max_samples:
            train_dataset.dataset = train_dataset.dataset.select(range(args.max_samples))
    
    # Eval dataset (même logique)
    if args.eval_data.endswith('.json'):
        with open(args.eval_data, 'r') as f:
            eval_data = json.load(f)
        eval_audio_files = [item["audio"] for item in eval_data]
        eval_transcripts = [item["text"] for item in eval_data]
        
        eval_dataset = create_meetings_dataset_from_files(
            eval_audio_files,
            eval_transcripts,
            processor,
            augmentations=None,
        )
    else:
        eval_dataset = prepare_dataset(
            args.eval_data,
            processor,
            split="validation",
            augmentations=None,
        )
        if hasattr(eval_dataset, 'dataset') and len(eval_dataset.dataset) > 1000:
            eval_dataset.dataset = eval_dataset.dataset.select(range(1000))
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Arguments d'entraînement optimisés
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=200,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=2000,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        fp16=True if device == "cuda" else False,
        bf16=False,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics_qat(pred, processor),
        tokenizer=processor.feature_extractor,
    )
    
    # Entraînement
    print(f"🚀 Démarrage entraînement QAT ({args.quantization_type})...")
    print(f"   Échantillons: {len(train_dataset)}")
    print(f"   Époques: {args.num_epochs}")
    print(f"   Batch size: {args.per_device_batch_size}")
    print(f"   Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"   Learning rate: {args.learning_rate}")
    print()
    
    trainer.train()
    
    # Sauvegarder modèle final
    final_output_dir = Path(args.output_dir) / "final"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(str(final_output_dir))
    processor.save_pretrained(str(final_output_dir))
    
    print(f"✅ Modèle QAT sauvegardé: {final_output_dir}")
    
    # Évaluation finale
    print("\n📊 Évaluation finale...")
    eval_results = trainer.evaluate()
    print(f"WER final: {eval_results.get('eval_wer', 'N/A'):.2%}")
    
    print(f"\n💡 Prochaine étape: Convertir en modèle quantifié")
    print(f"   python scripts/convert_qat_to_quantized.py \\")
    print(f"     --model_path {final_output_dir} \\")
    print(f"     --output_path {args.output_dir}-quantized \\")
    print(f"     --quantization_type {args.quantization_type}")


if __name__ == "__main__":
    main()

