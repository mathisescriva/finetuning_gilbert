#!/usr/bin/env python3
"""
Script pour générer automatiquement les transcripts d'un dataset audio sans transcripts.
Utilise Whisper pour créer des pseudo-labels (transcriptions automatiques).
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Dataset, DatasetDict
import librosa
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))


class AutoTranscriptGenerator:
    """Générateur de transcripts automatiques avec Whisper."""
    
    def __init__(
        self,
        model_name: str = "bofenghuang/whisper-large-v3-distil-fr-v0.2",
        device: str = None,
        batch_size: int = 1,
    ):
        """
        Args:
            model_name: Nom du modèle Whisper à utiliser
            device: Device (cuda/cpu), auto-détecté si None
            batch_size: Taille de batch (1 par défaut, car audio peut varier)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        print(f"Chargement du modèle {model_name}...")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Modèle chargé sur {self.device}")
    
    def transcribe_audio(self, audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Transcrit un audio avec Whisper.
        
        Args:
            audio_array: Array audio numpy
            sample_rate: Sample rate
        
        Returns:
            Dict avec 'text' et 'confidence' (si disponible)
        """
        # Préparer inputs
        inputs = self.processor(
            audio=audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self.device)
        
        # Générer transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"],
                max_length=448,
                num_beams=5,
                language="fr",
                task="transcribe",
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Décoder
        transcription = self.processor.batch_decode(
            generated_ids.sequences,
            skip_special_tokens=True,
        )[0]
        
        # Calculer confidence approximative depuis les scores
        confidence = None
        if hasattr(generated_ids, 'scores') and generated_ids.scores:
            # Moyenne des log probs (approximation)
            log_probs = []
            for score in generated_ids.scores:
                # Softmax pour obtenir probabilités
                probs = torch.softmax(score, dim=-1)
                # Probabilité du token choisi
                max_probs = torch.max(probs, dim=-1)[0]
                log_probs.append(max_probs.mean().item())
            
            if log_probs:
                # Moyenne des confidences par token
                confidence = np.mean(log_probs)
        
        return {
            "text": transcription,
            "confidence": confidence,
        }
    
    def transcribe_dataset(
        self,
        dataset,
        audio_column: str = "audio",
        sample_rate: int = 16000,
        max_samples: int = None,
        min_confidence: float = None,
        save_intermediate: bool = True,
        output_path: str = None,
    ) -> Dataset:
        """
        Transcrit tous les audios d'un dataset.
        
        Args:
            dataset: Dataset HuggingFace avec colonne audio
            audio_column: Nom de la colonne audio
            sample_rate: Sample rate cible
            max_samples: Limiter nombre d'échantillons (pour test)
            min_confidence: Filtrer par confidence minimale (optionnel)
            save_intermediate: Sauvegarder périodiquement
            output_path: Chemin pour sauvegarder résultats intermédiaires
        
        Returns:
            Dataset avec colonne 'text' ajoutée
        """
        print(f"Transcription de {len(dataset)} échantillons...")
        
        transcripts = []
        confidences = []
        failed_indices = []
        
        # Limiter si demandé
        dataset_to_process = dataset
        if max_samples and len(dataset) > max_samples:
            dataset_to_process = dataset.select(range(max_samples))
            print(f"  Limité à {max_samples} échantillons pour test")
        
        # Traiter chaque échantillon
        for idx, example in enumerate(tqdm(dataset_to_process, desc="Transcription")):
            try:
                # Extraire audio
                audio_data = example[audio_column]
                
                if audio_data is None:
                    print(f"  ⚠️  Échantillon {idx}: audio manquant")
                    transcripts.append("")
                    confidences.append(0.0)
                    failed_indices.append(idx)
                    continue
                
                # Charger audio si nécessaire
                if isinstance(audio_data, dict):
                    audio_array = audio_data["array"]
                    sr = audio_data.get("sampling_rate", sample_rate)
                elif isinstance(audio_data, str):
                    # Chemin vers fichier
                    audio_array, sr = librosa.load(audio_data, sr=sample_rate)
                else:
                    audio_array = audio_data
                    sr = sample_rate
                
                # Resample si nécessaire
                if sr != sample_rate:
                    audio_array = librosa.resample(
                        audio_array.astype(np.float32),
                        orig_sr=sr,
                        target_sr=sample_rate,
                    )
                
                # Normaliser
                if np.max(np.abs(audio_array)) > 0:
                    audio_array = audio_array / np.max(np.abs(audio_array))
                
                # Transcrire
                result = self.transcribe_audio(audio_array, sample_rate)
                
                transcripts.append(result["text"])
                confidences.append(result.get("confidence", 0.0))
                
                # Sauvegarde intermédiaire tous les 100 échantillons
                if save_intermediate and (idx + 1) % 100 == 0 and output_path:
                    self._save_intermediate(
                        dataset_to_process,
                        transcripts,
                        confidences,
                        idx + 1,
                        output_path,
                    )
                
            except Exception as e:
                print(f"  ❌ Erreur échantillon {idx}: {e}")
                transcripts.append("")
                confidences.append(0.0)
                failed_indices.append(idx)
        
        # Statistiques
        valid_transcripts = [t for t in transcripts if t]
        avg_confidence = np.mean([c for c in confidences if c > 0]) if confidences else 0.0
        
        print(f"\n📊 Statistiques:")
        print(f"  Total: {len(transcripts)}")
        print(f"  Réussis: {len(valid_transcripts)}")
        print(f"  Échoués: {len(failed_indices)}")
        print(f"  Confiance moyenne: {avg_confidence:.3f}")
        
        if failed_indices:
            print(f"  ⚠️  Indices échoués: {failed_indices[:10]}..." if len(failed_indices) > 10 else f"  ⚠️  Indices échoués: {failed_indices}")
        
        # Ajouter colonnes au dataset
        dataset_with_text = dataset_to_process.add_column("text", transcripts)
        dataset_with_text = dataset_with_text.add_column("transcription_confidence", confidences)
        dataset_with_text = dataset_with_text.add_column("auto_generated", [True] * len(transcripts))
        
        # Filtrer par confidence si demandé
        if min_confidence:
            before_filter = len(dataset_with_text)
            dataset_with_text = dataset_with_text.filter(
                lambda x: x["transcription_confidence"] >= min_confidence
            )
            print(f"  Filtré (confidence >= {min_confidence}): {before_filter} → {len(dataset_with_text)}")
        
        return dataset_with_text
    
    def _save_intermediate(
        self,
        dataset,
        transcripts,
        confidences,
        num_processed,
        output_path,
    ):
        """Sauvegarde intermédiaire."""
        try:
            temp_dataset = dataset.select(range(num_processed))
            temp_dataset = temp_dataset.add_column("text", transcripts[:num_processed])
            temp_dataset = temp_dataset.add_column("transcription_confidence", confidences[:num_processed])
            
            temp_path = Path(output_path) / f"intermediate_{num_processed}"
            temp_dataset.save_to_disk(str(temp_path))
        except Exception as e:
            print(f"  ⚠️  Impossible de sauvegarder intermédiaire: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Générer automatiquement les transcripts d'un dataset audio"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MEscriva/french-education-speech",
        help="Nom du dataset HuggingFace",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split du dataset à traiter",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bofenghuang/whisper-large-v3-distil-fr-v0.2",
        help="Modèle Whisper à utiliser",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Répertoire de sortie",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Nom du dataset de sortie (défaut: {dataset_name}_with_transcripts)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Nombre max d'échantillons (pour test rapide)",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=None,
        help="Confidence minimale pour garder transcript (0-1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu), auto si None",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Pousser le dataset sur HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Token HuggingFace (ou variable env HF_TOKEN)",
    )
    
    args = parser.parse_args()
    
    # Charger dataset
    print(f"📥 Chargement du dataset {args.dataset_name}...")
    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        print("💡 Essayez avec --split train ou vérifiez le nom du dataset")
        return
    
    print(f"✅ Dataset chargé: {len(dataset)} échantillons")
    
    # Vérifier colonnes
    print(f"   Colonnes: {dataset.column_names}")
    
    # Identifier colonne audio
    audio_column = None
    for col in ["audio", "path", "file"]:
        if col in dataset.column_names:
            audio_column = col
            break
    
    if not audio_column:
        print("⚠️  Colonne audio non trouvée, tentative avec 'audio'...")
        audio_column = "audio"
    
    print(f"   Colonne audio utilisée: {audio_column}")
    
    # Créer générateur
    generator = AutoTranscriptGenerator(
        model_name=args.model_name,
        device=args.device,
    )
    
    # Nom de sortie
    output_name = args.output_name or f"{args.dataset_name.replace('/', '_')}_with_transcripts"
    output_path = Path(args.output_dir) / output_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Générer transcripts
    print(f"\n🎙️  Génération des transcripts...")
    dataset_with_transcripts = generator.transcribe_dataset(
        dataset,
        audio_column=audio_column,
        max_samples=args.max_samples,
        min_confidence=args.min_confidence,
        save_intermediate=True,
        output_path=str(output_path),
    )
    
    # Sauvegarder
    print(f"\n💾 Sauvegarde dans {output_path}...")
    
    # Si dataset avait plusieurs splits, créer DatasetDict
    try:
        full_dataset = load_dataset(args.dataset_name)
        if isinstance(full_dataset, DatasetDict):
            # Mettre à jour le split traité
            full_dataset[args.split] = dataset_with_transcripts
            full_dataset.save_to_disk(str(output_path))
        else:
            dataset_with_transcripts.save_to_disk(str(output_path))
    except:
        dataset_with_transcripts.save_to_disk(str(output_path))
    
    print(f"✅ Dataset sauvegardé dans {output_path}")
    
    # Exporter aussi en JSON pour référence
    json_path = output_path / "transcripts.json"
    transcripts_list = [
        {
            "index": i,
            "text": dataset_with_transcripts[i]["text"],
            "confidence": dataset_with_transcripts[i].get("transcription_confidence", 0.0),
        }
        for i in range(len(dataset_with_transcripts))
    ]
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcripts_list, f, indent=2, ensure_ascii=False)
    print(f"✅ Transcripts JSON sauvegardés dans {json_path}")
    
    # Pousser sur Hub si demandé
    if args.push_to_hub:
        print(f"\n🚀 Pousse vers HuggingFace Hub...")
        try:
            token = args.hub_token or os.getenv("HF_TOKEN")
            if not token:
                print("⚠️  Token HuggingFace non fourni, skip push_to_hub")
            else:
                dataset_with_transcripts.push_to_hub(
                    output_name,
                    token=token,
                )
                print(f"✅ Dataset poussé sur Hub: {output_name}")
        except Exception as e:
            print(f"❌ Erreur lors du push: {e}")
    
    print(f"\n{'='*60}")
    print("✅ GÉNÉRATION TERMINÉE")
    print(f"{'='*60}")
    print(f"\n📁 Dataset avec transcripts: {output_path}")
    print(f"\n💡 Pour utiliser ce dataset pour fine-tuning:")
    print(f"   python scripts/fine_tune_meetings.py \\")
    print(f"     --train_data {output_path} \\")
    print(f"     --eval_data {output_path} \\")
    print(f"     --phase 1")


if __name__ == "__main__":
    import os
    main()

