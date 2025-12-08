#!/usr/bin/env python3
"""
Script de fine-tuning Whisper sur données de réunions.
Supporte fine-tuning full et LoRA.
"""

import argparse
import yaml
import json
from pathlib import Path
import torch
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
from src.model.whisper_lora import create_lora_whisper_from_config
from src.evaluation.metrics import compute_wer
from src.training.trainer import DataCollatorSpeechSeq2SeqWithPadding


def load_data_config(config_path: str):
    """Charge la config de données."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get("data", {})


def load_training_config(config_path: str):
    """Charge la config d'entraînement."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get("training", {})


def compute_metrics(pred, processor, metric_key_prefix: str = "eval"):
    """Calcule WER comme métrique d'évaluation."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Remplacer -100 (ignored) par pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Décoder
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Calculer WER
    wer = compute_wer(label_str, pred_str)
    
    return {f"{metric_key_prefix}_wer": wer}


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Whisper pour réunions")
    parser.add_argument(
        "--model_name",
        type=str,
        default="bofenghuang/whisper-large-v3-distil-fr-v0.2",
        help="Modèle de base",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Chemin vers données d'entraînement (JSON ou HuggingFace dataset)",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        required=True,
        help="Chemin vers données d'évaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/models/whisper-meetings",
        help="Répertoire de sortie",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="config/training_config.yaml",
        help="Config d'entraînement",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/model_config.yaml",
        help="Config modèle",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Utiliser LoRA au lieu de fine-tuning full",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3"],
        default="1",
        help="Phase d'entraînement (1=encoder frozen, 2=full, 3=LoRA)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Reprendre depuis un checkpoint",
    )
    
    args = parser.parse_args()
    
    # Charger configs
    training_config_all = load_training_config(args.training_config)
    phase_config = training_config_all.get(f"phase{args.phase}", {})
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Sélectionner config selon phase
    if args.use_lora and args.phase == "3":
        phase_config = training_config_all.get("phase3_lora", {})
        if not phase_config.get("enabled", False):
            print("Phase 3 LoRA non activée dans la config, activation...")
            phase_config["enabled"] = True
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Charger modèle et processor
    print(f"Chargement du modèle {args.model_name}...")
    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language="fr",
        task="transcribe",
    )
    
    if args.use_lora and args.phase == "3":
        # Créer modèle avec LoRA
        lora_config_dict = model_config.get("lora", {})
        model = create_lora_whisper_from_config(
            args.model_name,
            lora_kwargs=lora_config_dict,
        )
        print("Modèle avec LoRA créé")
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
        
        # Freeze encoder si phase 1
        if args.phase == "1" and phase_config.get("freeze_encoder", False):
            print("Encoder gelé")
            for param in model.model.encoder.parameters():
                param.requires_grad = False
    
    # Charger données
    print(f"Chargement des données depuis {args.train_data}...")
    
    # Support pour JSON local ou HuggingFace dataset
    if args.train_data.endswith('.json'):
        with open(args.train_data, 'r') as f:
            train_data = json.load(f)
        audio_files = [item["audio"] for item in train_data]
        transcripts = [item["text"] for item in train_data]
        train_dataset = create_meetings_dataset_from_files(
            audio_files,
            transcripts,
            processor,
            augmentations=create_augmentation_pipeline(training_config_all.get("data", {})),
        )
    else:
        # HuggingFace dataset
        train_dataset = prepare_dataset(
            args.train_data,
            processor,
            split="train",
            augmentations=create_augmentation_pipeline(training_config_all.get("data", {})),
        )
    
    # Données eval
    if args.eval_data.endswith('.json'):
        with open(args.eval_data, 'r') as f:
            eval_data = json.load(f)
        eval_audio_files = [item["audio"] for item in eval_data]
        eval_transcripts = [item["text"] for item in eval_data]
        eval_dataset = create_meetings_dataset_from_files(
            eval_audio_files,
            eval_transcripts,
            processor,
            augmentations=None,  # Pas d'augmentation pour eval
        )
    else:
        eval_dataset = prepare_dataset(
            args.eval_data,
            processor,
            split="validation",
            augmentations=None,
        )
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Arguments d'entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=phase_config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=phase_config.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=phase_config.get("gradient_accumulation_steps", 4),
        learning_rate=phase_config.get("learning_rate", 1e-5),
        warmup_steps=phase_config.get("warmup_steps", 500),
        num_train_epochs=phase_config.get("num_epochs", 3),
        evaluation_strategy="steps",
        eval_steps=phase_config.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=phase_config.get("save_steps", 1000),
        logging_steps=phase_config.get("logging_steps", 100),
        load_best_model_at_end=True,
        metric_for_best_model="eval_wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        fp16=training_config_all.get("fp16", True),
        bf16=training_config_all.get("bf16", False),
        lr_scheduler_type=phase_config.get("lr_scheduler_type", "cosine"),
        weight_decay=phase_config.get("weight_decay", 0.01),
        max_grad_norm=phase_config.get("max_grad_norm", 1.0),
        seed=training_config_all.get("seed", 42),
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        tokenizer=processor.feature_extractor,
    )
    
    # Entraînement
    print(f"Démarrage de l'entraînement (Phase {args.phase})...")
    
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()
    
    # Sauvegarder modèle final
    final_output_dir = Path(args.output_dir) / "final"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(str(final_output_dir))
    processor.save_pretrained(str(final_output_dir))
    
    print(f"Modèle sauvegardé dans {final_output_dir}")
    
    # Évaluation finale
    print("Évaluation finale...")
    eval_results = trainer.evaluate()
    print(f"WER final: {eval_results.get('eval_wer', 'N/A')}%")


if __name__ == "__main__":
    main()

