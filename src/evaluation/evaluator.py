"""
Évaluateur complet pour modèles Whisper.
Gère transcription, calcul de métriques, et benchmarking.
"""

import torch
import time
from typing import List, Dict, Optional, Tuple
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path
import numpy as np
import librosa

from .metrics import compute_detailed_metrics


class WhisperEvaluator:
    """Évaluateur pour modèles Whisper."""
    
    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        chunk_length_s: int = 30,
        beam_size: int = 5,
    ):
        """
        Args:
            model: Modèle Whisper
            processor: WhisperProcessor
            device: Device pour inférence
            chunk_length_s: Longueur des chunks en secondes
            beam_size: Taille du beam search
        """
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.chunk_length_s = chunk_length_s
        self.beam_size = beam_size
        
        # Mode évaluation
        self.model.eval()
    
    def transcribe_audio(
        self,
        audio_path: str,
        return_timings: bool = False,
    ) -> str:
        """
        Transcrit un fichier audio.
        
        Args:
            audio_path: Chemin vers fichier audio
            return_timings: Retourner aussi les timings
        
        Returns:
            Texte transcrit (et timings si demandé)
        """
        # Charger audio
        audio_array, sample_rate = librosa.load(audio_path, sr=16000)
        duration = len(audio_array) / sample_rate
        
        # Traiter avec processor
        inputs = self.processor(
            audio=audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self.device)
        
        # Générer transcription
        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"],
                max_length=448,  # Longueur max Whisper
                num_beams=self.beam_size,
                language="fr",
                task="transcribe",
            )
        
        inference_time = time.time() - start_time
        
        # Décoder
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        
        if return_timings:
            return transcription, {
                "inference_time": inference_time,
                "audio_duration": duration,
                "real_time_factor": inference_time / duration if duration > 0 else 0,
            }
        return transcription
    
    def evaluate_on_dataset(
        self,
        audio_files: List[str],
        references: List[str],
        entity_list: Optional[set] = None,
        return_individual: bool = False,
    ) -> Dict:
        """
        Évalue le modèle sur un dataset.
        
        Args:
            audio_files: Liste de chemins audio
            references: Liste de transcripts de référence
            entity_list: Set d'entités connues
            return_individual: Retourner métriques individuelles
        
        Returns:
            Dict avec métriques globales et optionnellement individuelles
        """
        hypotheses = []
        timings = []
        
        print(f"Évaluation sur {len(audio_files)} fichiers...")
        for i, audio_path in enumerate(audio_files):
            if (i + 1) % 10 == 0:
                print(f"  Traité {i+1}/{len(audio_files)}...")
            
            transcription, timing = self.transcribe_audio(
                audio_path,
                return_timings=True,
            )
            hypotheses.append(transcription)
            timings.append(timing)
        
        # Calculer métriques
        metrics = compute_detailed_metrics(
            references,
            hypotheses,
            entity_list=entity_list,
        )
        
        # Ajouter métriques de performance
        avg_rttf = np.mean([t["real_time_factor"] for t in timings])
        avg_inference_time = np.mean([t["inference_time"] for t in timings])
        total_audio_duration = sum([t["audio_duration"] for t in timings])
        
        metrics["performance"] = {
            "avg_real_time_factor": avg_rttf,
            "avg_inference_time_per_file": avg_inference_time,
            "total_audio_duration": total_audio_duration,
            "total_inference_time": sum([t["inference_time"] for t in timings]),
            "latency_per_minute": avg_inference_time / (total_audio_duration / 60) if total_audio_duration > 0 else 0,
        }
        
        result = {
            "metrics": metrics,
            "hypotheses": hypotheses,
            "references": references,
        }
        
        if return_individual:
            result["individual_timings"] = timings
        
        return result
    
    def benchmark_inference(
        self,
        audio_path: str,
        num_runs: int = 5,
    ) -> Dict:
        """
        Benchmark performance d'inférence.
        
        Args:
            audio_path: Fichier audio de test
            num_runs: Nombre de runs pour moyenne
        
        Returns:
            Dict avec métriques de performance
        """
        times = []
        audio_array, sample_rate = librosa.load(audio_path, sr=16000)
        duration = len(audio_array) / sample_rate
        
        inputs = self.processor(
            audio=audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self.device)
        
        # Warmup
        with torch.no_grad():
            _ = self.model.generate(
                inputs["input_features"],
                max_length=448,
                num_beams=self.beam_size,
            )
        
        # Benchmark
        for _ in range(num_runs):
            torch.cuda.synchronize() if self.device == "cuda" else None
            start = time.time()
            
            with torch.no_grad():
                _ = self.model.generate(
                    inputs["input_features"],
                    max_length=448,
                    num_beams=self.beam_size,
                )
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return {
            "avg_inference_time": avg_time,
            "std_inference_time": std_time,
            "audio_duration": duration,
            "real_time_factor": avg_time / duration if duration > 0 else 0,
            "throughput_seconds_per_audio_second": avg_time / duration if duration > 0 else 0,
        }
    
    def get_model_memory_usage(self) -> Dict:
        """
        Mesure l'utilisation mémoire du modèle.
        
        Returns:
            Dict avec VRAM/RAM utilisée
        """
        if self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            return {
                "gpu_memory_allocated_gb": memory_allocated,
                "gpu_memory_reserved_gb": memory_reserved,
            }
        else:
            # Estimation CPU (approximative)
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024**2
            return {
                "cpu_memory_mb": memory_mb,
                "cpu_memory_gb": memory_mb / 1024,
            }

