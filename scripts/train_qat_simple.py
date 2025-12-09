#!/usr/bin/env python3
"""
Script QAT simplifié basé sur les exemples HuggingFace
Utilise un dataset limité chargé en mémoire (pas de streaming complexe)
"""

import argparse
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset
from jiwer import wer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.training.trainer import DataCollatorSpeechSeq2SeqWithPadding

# Configurer temp
import os
os.environ["TMPDIR"] = os.environ.get("TMPDIR", "/workspace/tmp")
os.environ["TEMP"] = os.environ.get("TEMP", "/workspace/tmp")
os.environ["TMP"] = os.environ.get("TMP", "/workspace/tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)


def prepare_dataset_simple(dataset_name, config, split, processor, max_samples=10000):
    """Charge et prépare un dataset de manière simple (limité en mémoire)."""
    print(f"📊 Chargement {dataset_name} ({config}) - split: {split}")
    
    # Charger dataset
    dataset = load_dataset(dataset_name, config, split=split, streaming=False)
    
    # Limiter immédiatement
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    print(f"✅ {len(dataset)} échantillons chargés")
    
    # Fonction de preprocessing
    def prepare_example(example):
        audio = example["audio"]
        text = example.get("text", "")
        
        # Vérifier format audio
        if isinstance(audio, dict):
            audio_array = audio.get("array", None)
            sr = audio.get("sampling_rate", 16000)
        else:
            audio_array = audio
            sr = 16000
        
        if audio_array is None:
            return None
        
        # Preprocess audio avec processor
        inputs = processor(
            audio=audio_array,
            sampling_rate=sr,
            text=text,
            return_tensors="pt",
        )
        
        # Convertir en format attendu (numpy pour compatibilité)
        input_features = inputs.input_features.squeeze(0).numpy()
        labels = inputs.input_ids.squeeze(0).numpy()
        
        return {
            "input_features": input_features,
            "labels": labels,
        }
    
    # Appliquer preprocessing
    print("🔧 Preprocessing...")
    dataset = dataset.map(
        prepare_example,
        remove_columns=[col for col in dataset.column_names],
        num_proc=2,  # Réduire workers pour éviter problèmes mémoire
        desc="Preprocessing audio",
    )
    
    # Filtrer None (si certains exemples ont échoué)
    dataset = dataset.filter(lambda x: x["input_features"] is not None)
    
    return dataset


def compute_metrics(eval_pred, processor):
    """Calcule WER."""
    predictions, labels = eval_pred
    
    # Remplacer -100 par pad_token_id
    labels[labels == -100] = processor.tokenizer.pad_token_id
    
    # Décoder
    pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)
    
    # Calculer WER
    wer_score = wer(label_str, pred_str)
    
    return {"wer": wer_score}


def main():
    parser = argparse.ArgumentParser(description="QAT simple pour Whisper")
    parser.add_argument("--base_model", default="bofenghuang/whisper-large-v3-distil-fr-v0.2")
    parser.add_argument("--output_dir", default="outputs/models/gilbert-whisper-qat-int8")
    parser.add_argument("--max_samples", type=int, default=10000, help="Max samples (limite mémoire)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Époques")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 QAT SIMPLIFIÉ - Whisper")
    print("="*60)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎮 Device: {device}")
    
    # Charger modèle et processor
    print(f"📦 Chargement modèle: {args.base_model}")
    processor = WhisperProcessor.from_pretrained(args.base_model, language="fr", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    
    # Activer fake quantization (simplifié)
    print("🔧 Activation fake quantization...")
    model.train()
    # Note: Pour une vraie QAT, on devrait utiliser torch.quantization.prepare_qat
    # Mais pour simplifier, on va juste entraîner le modèle normalement
    # La quantization sera faite après avec PTQ
    print("⚠️  Note: Fake quantization simplifiée (entraînement normal)")
    print("   Pour vraie QAT, voir scripts/train_qat.py")
    
    # Charger dataset (MLS français, limité)
    print("\n📊 Chargement dataset...")
    train_dataset = prepare_dataset_simple(
        "facebook/multilingual_librispeech",
        "french",
        "train",
        processor,
        max_samples=args.max_samples,
    )
    
    eval_dataset = prepare_dataset_simple(
        "facebook/multilingual_librispeech",
        "french",
        "dev",
        processor,
        max_samples=min(1000, args.max_samples // 10),
    )
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        num_train_epochs=args.num_epochs,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=device == "cuda",
        report_to=[],
        push_to_hub=False,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        processing_class=processor.feature_extractor,
    )
    
    # Entraînement
    print("\n🚀 Démarrage entraînement...")
    trainer.train()
    
    # Sauvegarder
    print(f"\n💾 Sauvegarde dans {args.output_dir}")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    print("\n✅ Entraînement terminé !")
    print(f"\n💡 Prochaine étape: Quantifier avec PTQ")
    print(f"   python scripts/quantize_ptq.py --model {args.output_dir}")


if __name__ == "__main__":
    main()

