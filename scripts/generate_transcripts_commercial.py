#!/usr/bin/env python3
"""
Script pour générer automatiquement les transcripts avec des services commerciaux
(AssemblyAI, Deepgram, Rev.ai, etc.) - meilleure qualité que Whisper gratuit.
"""

import argparse
import json
import os
import time
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Dataset
from typing import Dict, List, Optional
import librosa
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))


class CommercialTranscriptService:
    """Interface pour services de transcription commerciaux."""
    
    def __init__(self, service: str = "assemblyai", api_key: str = None):
        """
        Args:
            service: "assemblyai", "deepgram", "rev", "azure", "google"
            api_key: Clé API (ou variable d'environnement)
        """
        self.service = service.lower()
        self.api_key = api_key or os.getenv(f"{self.service.upper()}_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                f"API key non fournie. Définissez {self.service.upper()}_API_KEY "
                "ou passez --api_key"
            )
        
        self._init_service()
    
    def _init_service(self):
        """Initialise le service spécifique."""
        if self.service == "assemblyai":
            try:
                import assemblyai as aai
                aai.settings.api_key = self.api_key
                self.client = aai
                print("✅ AssemblyAI initialisé")
            except ImportError:
                raise ImportError("Installez assemblyai: pip install assemblyai")
        
        elif self.service == "deepgram":
            try:
                from deepgram import DeepgramClient, PrerecordedOptions, FileSource
                self.client = DeepgramClient(self.api_key)
                self.options = PrerecordedOptions(
                    model="nova-2",
                    language="fr",
                    smart_format=True,
                    punctuate=True,
                )
                print("✅ Deepgram initialisé")
            except ImportError:
                raise ImportError("Installez deepgram-sdk: pip install deepgram-sdk")
        
        elif self.service == "azure":
            try:
                from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig
                self.speech_config = SpeechConfig(subscription=self.api_key, region=os.getenv("AZURE_REGION", "francecentral"))
                self.speech_config.speech_recognition_language = "fr-FR"
                print("✅ Azure Speech Services initialisé")
            except ImportError:
                raise ImportError("Installez azure-cognitiveservices-speech: pip install azure-cognitiveservices-speech")
        
        elif self.service == "google":
            try:
                from google.cloud import speech_v1
                import google.auth
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.api_key  # Path to JSON key
                self.client = speech_v1.SpeechClient()
                print("✅ Google Cloud Speech-to-Text initialisé")
            except ImportError:
                raise ImportError("Installez google-cloud-speech: pip install google-cloud-speech")
        
        else:
            raise ValueError(f"Service non supporté: {self.service}")
    
    def transcribe_audio_file(self, audio_path: str) -> Dict:
        """
        Transcrit un fichier audio.
        
        Returns:
            Dict avec 'text' et 'confidence' (si disponible)
        """
        if self.service == "assemblyai":
            return self._transcribe_assemblyai(audio_path)
        elif self.service == "deepgram":
            return self._transcribe_deepgram(audio_path)
        elif self.service == "azure":
            return self._transcribe_azure(audio_path)
        elif self.service == "google":
            return self._transcribe_google(audio_path)
    
    def transcribe_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Transcrit un array audio (sauvegarde temporaire puis transcription).
        """
        import tempfile
        import soundfile as sf
        
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_array, sample_rate)
            tmp_path = tmp_file.name
        
        try:
            result = self.transcribe_audio_file(tmp_path)
        finally:
            # Nettoyer
            os.unlink(tmp_path)
        
        return result
    
    def _transcribe_assemblyai(self, audio_path: str) -> Dict:
        """Transcription avec AssemblyAI."""
        import assemblyai as aai
        
        # Upload fichier
        transcript = aai.Transcriber().transcribe(audio_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            return {
                "text": "",
                "confidence": 0.0,
                "error": transcript.error,
            }
        
        # Confiance moyenne (si disponible)
        confidence = None
        if transcript.confidence is not None:
            confidence = transcript.confidence
        
        return {
            "text": transcript.text or "",
            "confidence": confidence or 0.9,  # AssemblyAI est généralement fiable
            "words": [{"word": w.text, "start": w.start, "end": w.end} for w in transcript.words] if hasattr(transcript, 'words') else None,
        }
    
    def _transcribe_deepgram(self, audio_path: str) -> Dict:
        """Transcription avec Deepgram."""
        from deepgram import PrerecordedOptions, FileSource
        
        with open(audio_path, "rb") as audio_file:
            payload: FileSource = {
                "buffer": audio_file,
            }
            
            response = self.client.listen.rest.v("1").transcribe_file(
                payload, self.options
            )
            
            if response.results is None or not response.results.channels:
                return {"text": "", "confidence": 0.0}
            
            # Extraire texte
            transcript = response.results.channels[0].alternatives[0].transcript
            confidence = response.results.channels[0].alternatives[0].confidence
            
            return {
                "text": transcript,
                "confidence": confidence or 0.9,
            }
    
    def _transcribe_azure(self, audio_path: str) -> Dict:
        """Transcription avec Azure Speech Services."""
        from azure.cognitiveservices.speech import SpeechRecognizer, AudioConfig, ResultReason
        
        audio_config = AudioConfig(filename=audio_path)
        recognizer = SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )
        
        result = recognizer.recognize_once()
        
        if result.reason == ResultReason.RecognizedSpeech:
            return {
                "text": result.text,
                "confidence": 0.9,  # Azure ne retourne pas toujours confidence
            }
        else:
            return {
                "text": "",
                "confidence": 0.0,
                "error": result.reason,
            }
    
    def _transcribe_google(self, audio_path: str) -> Dict:
        """Transcription avec Google Cloud Speech-to-Text."""
        from google.cloud import speech_v1
        
        # Lire audio
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()
        
        audio = speech_v1.RecognitionAudio(content=content)
        config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="fr-FR",
            enable_automatic_punctuation=True,
        )
        
        response = self.client.recognize(config=config, audio=audio)
        
        if not response.results:
            return {"text": "", "confidence": 0.0}
        
        # Concaténer tous les résultats
        text_parts = []
        confidences = []
        
        for result in response.results:
            alternative = result.alternatives[0]
            text_parts.append(alternative.transcript)
            if alternative.confidence > 0:
                confidences.append(alternative.confidence)
        
        return {
            "text": " ".join(text_parts),
            "confidence": np.mean(confidences) if confidences else 0.9,
        }


def generate_transcripts_commercial(
    dataset,
    service: str,
    api_key: str,
    audio_column: str = "audio",
    max_samples: int = None,
    output_path: str = None,
    delay_between_requests: float = 0.5,
) -> Dataset:
    """
    Génère transcripts avec service commercial.
    
    Args:
        dataset: Dataset HuggingFace
        service: Service à utiliser
        api_key: Clé API
        audio_column: Colonne audio
        max_samples: Limiter nombre d'échantillons
        output_path: Chemin de sortie
        delay_between_requests: Délai entre requêtes (rate limiting)
    """
    print(f"🚀 Utilisation du service {service.upper()} pour transcription...")
    
    # Initialiser service
    transcript_service = CommercialTranscriptService(service, api_key)
    
    # Limiter si demandé
    dataset_to_process = dataset
    if max_samples and len(dataset) > max_samples:
        dataset_to_process = dataset.select(range(max_samples))
        print(f"  Limité à {max_samples} échantillons")
    
    transcripts = []
    confidences = []
    errors = []
    
    # Traiter chaque échantillon
    for idx, example in enumerate(tqdm(dataset_to_process, desc="Transcription")):
        try:
            # Extraire audio
            audio_data = example[audio_column]
            
            if audio_data is None:
                transcripts.append("")
                confidences.append(0.0)
                errors.append(f"Index {idx}: audio manquant")
                continue
            
            # Préparer audio
            if isinstance(audio_data, dict):
                # Dataset HuggingFace avec audio chargé
                audio_array = audio_data["array"]
                sr = audio_data.get("sampling_rate", 16000)
                
                # Transcrire array
                result = transcript_service.transcribe_audio_array(audio_array, sr)
                
            elif isinstance(audio_data, str):
                # Chemin vers fichier
                result = transcript_service.transcribe_audio_file(audio_data)
            else:
                # Array direct
                result = transcript_service.transcribe_audio_array(audio_data)
            
            transcripts.append(result.get("text", ""))
            confidences.append(result.get("confidence", 0.0))
            
            if "error" in result:
                errors.append(f"Index {idx}: {result['error']}")
            
            # Rate limiting
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)
            
        except Exception as e:
            print(f"  ❌ Erreur échantillon {idx}: {e}")
            transcripts.append("")
            confidences.append(0.0)
            errors.append(f"Index {idx}: {str(e)}")
    
    # Statistiques
    valid_transcripts = [t for t in transcripts if t]
    avg_confidence = np.mean([c for c in confidences if c > 0]) if confidences else 0.0
    
    print(f"\n📊 Statistiques:")
    print(f"  Total: {len(transcripts)}")
    print(f"  Réussis: {len(valid_transcripts)}")
    print(f"  Échoués: {len(errors)}")
    print(f"  Confiance moyenne: {avg_confidence:.3f}")
    
    if errors:
        print(f"  ⚠️  Erreurs: {len(errors)}")
        if len(errors) <= 10:
            for err in errors:
                print(f"    - {err}")
    
    # Ajouter colonnes
    dataset_with_text = dataset_to_process.add_column("text", transcripts)
    dataset_with_text = dataset_with_text.add_column("transcription_confidence", confidences)
    dataset_with_text = dataset_with_text.add_column("transcription_service", [service] * len(transcripts))
    dataset_with_text = dataset_with_text.add_column("auto_generated", [True] * len(transcripts))
    
    return dataset_with_text


def main():
    parser = argparse.ArgumentParser(
        description="Générer transcripts avec services commerciaux (AssemblyAI, Deepgram, etc.)"
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
        help="Split du dataset",
    )
    parser.add_argument(
        "--service",
        type=str,
        choices=["assemblyai", "deepgram", "azure", "google"],
        default="assemblyai",
        help="Service de transcription à utiliser",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Clé API (ou variable d'environnement {SERVICE}_API_KEY)",
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
        help="Nom du dataset de sortie",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Nombre max d'échantillons (pour test)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Délai entre requêtes (rate limiting, secondes)",
    )
    
    args = parser.parse_args()
    
    # Charger dataset
    print(f"📥 Chargement du dataset {args.dataset_name}...")
    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return
    
    print(f"✅ Dataset chargé: {len(dataset)} échantillons")
    
    # Identifier colonne audio
    audio_column = "audio" if "audio" in dataset.column_names else dataset.column_names[0]
    print(f"   Colonne audio: {audio_column}")
    
    # Générer transcripts
    dataset_with_transcripts = generate_transcripts_commercial(
        dataset,
        service=args.service,
        api_key=args.api_key,
        audio_column=audio_column,
        max_samples=args.max_samples,
        delay_between_requests=args.delay,
    )
    
    # Sauvegarder
    output_name = args.output_name or f"{args.dataset_name.replace('/', '_')}_transcribed_{args.service}"
    output_path = Path(args.output_dir) / output_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset_with_transcripts.save_to_disk(str(output_path))
    print(f"✅ Dataset sauvegardé dans {output_path}")
    
    # Export JSON
    json_path = output_path / "transcripts.json"
    transcripts_list = [
        {
            "index": i,
            "text": dataset_with_transcripts[i]["text"],
            "confidence": dataset_with_transcripts[i]["transcription_confidence"],
            "service": args.service,
        }
        for i in range(len(dataset_with_transcripts))
    ]
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcripts_list, f, indent=2, ensure_ascii=False)
    print(f"✅ Transcripts JSON sauvegardés dans {json_path}")
    
    print(f"\n{'='*60}")
    print("✅ GÉNÉRATION TERMINÉE")
    print(f"{'='*60}")
    print(f"\n💡 Pour fine-tuning:")
    print(f"   python scripts/fine_tune_meetings.py \\")
    print(f"     --train_data {output_path} \\")
    print(f"     --eval_data {output_path} \\")
    print(f"     --phase 1")


if __name__ == "__main__":
    main()

