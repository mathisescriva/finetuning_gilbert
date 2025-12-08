"""
Composants additionnels pour l'entraînement Whisper.
"""

import torch
from transformers import WhisperProcessor
from typing import Dict, List


class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator pour Whisper qui gère le padding correctement.
    """
    
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate une batch de features.
        
        Args:
            features: Liste de dicts avec 'input_features', 'labels', etc.
        
        Returns:
            Dict avec batch collated
        """
        # Séparer input_features et labels
        input_features = [f["input_features"] for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        # Pad input_features
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )
        
        # Remplacer padding token par -100 (ignored dans loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # Si decoder_input_ids pas présent, utiliser labels (shifted)
        if "decoder_input_ids" not in labels_batch:
            labels_batch["decoder_input_ids"] = labels
        
        batch["labels"] = labels
        
        return batch

