"""
Dataset personnalisé pour l'entraînement Whisper sur données de réunions.
Gère le chargement audio, preprocessing, et préparation pour l'entraînement.
"""

import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import IterableDataset
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Union
import numpy as np
from transformers import WhisperProcessor


class MeetingsDataset(IterableDataset):
    """
    Dataset itérable pour les données de réunions.
    Optimisé pour gérer de grandes quantités de données audio.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        processor: WhisperProcessor,
        augmentations=None,
        max_duration: float = 30.0,
        min_duration: float = 1.0,
        sample_rate: int = 16000,
    ):
        """
        Args:
            dataset: Dataset HuggingFace contenant 'audio' et 'text'
            processor: WhisperProcessor pour tokenization
            augmentations: Pipeline d'augmentations audio (optionnel)
            max_duration: Durée maximale en secondes
            min_duration: Durée minimale en secondes
            sample_rate: Sample rate cible (16kHz pour Whisper)
        """
        self.dataset = dataset
        self.processor = processor
        self.augmentations = augmentations
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.sample_rate = sample_rate
    
    def __iter__(self):
        """Itère sur le dataset avec preprocessing et augmentation."""
        for item in self.dataset:
            # Charger audio
            audio = item["audio"]
            text = item.get("text", "")
            
            # Filtrer par durée
            if audio is None or "array" not in audio:
                continue
                
            duration = len(audio["array"]) / audio.get("sampling_rate", self.sample_rate)
            if duration < self.min_duration or duration > self.max_duration:
                continue
            
            # Resample si nécessaire
            audio_array = audio["array"]
            sr = audio.get("sampling_rate", self.sample_rate)
            
            if sr != self.sample_rate:
                audio_array = librosa.resample(
                    audio_array.astype(np.float32),
                    orig_sr=sr,
                    target_sr=self.sample_rate
                )
            
            # Normalisation
            audio_array = audio_array.astype(np.float32)
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Appliquer augmentations si disponibles
            if self.augmentations is not None:
                audio_array = self.augmentations(audio_array, self.sample_rate)
            
            # Traiter avec le processor Whisper
            inputs = self.processor(
                audio=audio_array,
                sampling_rate=self.sample_rate,
                text=text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_duration * self.sample_rate,
            )
            
            # Préparer les inputs pour l'entraînement
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            
            # Les labels sont les input_ids du texte
            if "labels" not in inputs:
                inputs["labels"] = inputs["input_ids"].clone()
            
            yield inputs
    
    def __len__(self):
        return len(self.dataset)


def prepare_dataset(
    dataset_path: str,
    processor: WhisperProcessor,
    split: str = "train",
    augmentations=None,
    max_duration: float = 30.0,
    min_duration: float = 1.0,
    streaming: bool = False,
) -> MeetingsDataset:
    """
    Prépare un dataset pour l'entraînement.
    
    Args:
        dataset_path: Chemin vers le dataset (HuggingFace hub ou local)
        processor: WhisperProcessor
        split: Split à utiliser
        augmentations: Pipeline d'augmentations
        max_duration: Durée maximale
        min_duration: Durée minimale
        streaming: Utiliser le streaming (pour datasets très larges)
    
    Returns:
        MeetingsDataset configuré
    """
    from datasets import load_dataset
    
    # Charger le dataset
    if streaming:
        dataset = load_dataset(dataset_path, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_path, split=split)
    
    return MeetingsDataset(
        dataset=dataset,
        processor=processor,
        augmentations=augmentations,
        max_duration=max_duration,
        min_duration=min_duration,
    )


def create_meetings_dataset_from_files(
    audio_files: List[str],
    transcripts: List[str],
    processor: WhisperProcessor,
    augmentations=None,
    max_duration: float = 30.0,
    min_duration: float = 1.0,
    sample_rate: int = 16000,
) -> MeetingsDataset:
    """
    Crée un dataset à partir de fichiers audio locaux et transcripts.
    
    Args:
        audio_files: Liste de chemins vers fichiers audio
        transcripts: Liste de transcripts correspondants
        processor: WhisperProcessor
        augmentations: Pipeline d'augmentations
        max_duration: Durée maximale
        min_duration: Durée minimale
        sample_rate: Sample rate cible
    
    Returns:
        MeetingsDataset
    """
    # Créer un dataset HuggingFace temporaire
    data = []
    for audio_path, text in zip(audio_files, transcripts):
        data.append({"audio_path": audio_path, "text": text})
    
    dataset = Dataset.from_list(data)
    
    # Fonction pour charger l'audio
    def load_audio(example):
        audio_array, sr = librosa.load(example["audio_path"], sr=sample_rate)
        return {
            "audio": {
                "array": audio_array,
                "sampling_rate": sr,
            },
            "text": example["text"],
        }
    
    dataset = dataset.map(load_audio, remove_columns=["audio_path"])
    
    return MeetingsDataset(
        dataset=dataset,
        processor=processor,
        augmentations=augmentations,
        max_duration=max_duration,
        min_duration=min_duration,
        sample_rate=sample_rate,
    )

