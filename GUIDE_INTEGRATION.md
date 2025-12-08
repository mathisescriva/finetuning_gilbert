# Guide d'Intégration : Modèle Whisper Spécialisé Réunions

## Vue d'ensemble

Ce guide explique comment intégrer le modèle fine-tuné dans votre application de compte-rendu de réunion.

## 1. Installation

### Prérequis

```bash
pip install -r requirements.txt
```

### Installation spécifique selon backend

**Option 1 : Transformers (PyTorch) - Recommandé pour développement**
```bash
pip install transformers torch torchaudio
```

**Option 2 : Faster-Whisper (CTranslate2) - Recommandé pour production GPU**
```bash
pip install faster-whisper
```

**Option 3 : Whisper.cpp - Recommandé pour CPU/Edge**
```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make
```

## 2. Chargement du Modèle

### 2.1 Avec Transformers (PyTorch)

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Charger modèle et processor
model_name = "path/to/your/fine-tuned-model"  # ou nom HuggingFace
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
```

### 2.2 Avec Faster-Whisper (Recommandé Production)

```python
from faster_whisper import WhisperModel

model_path = "path/to/your/fine-tuned-model"
model = WhisperModel(model_path, device="cuda", compute_type="float16")
# ou pour CPU: model = WhisperModel(model_path, device="cpu", compute_type="int8")
```

### 2.3 Modèle Quantifié (Int8)

```python
from transformers import WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

model_path = "path/to/quantized-model"
processor = WhisperProcessor.from_pretrained(model_path)
model = ORTModelForSpeechSeq2Seq.from_pretrained(model_path)
```

## 3. Transcription Audio

### 3.1 Transcription Simple (Transformers)

```python
import librosa

def transcribe_audio(audio_path: str) -> str:
    # Charger audio
    audio_array, sample_rate = librosa.load(audio_path, sr=16000)
    
    # Préparer inputs
    inputs = processor(
        audio=audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).to(device)
    
    # Générer transcription
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"],
            max_length=448,
            num_beams=5,
            language="fr",
            task="transcribe",
        )
    
    # Décoder
    transcription = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]
    
    return transcription
```

### 3.2 Transcription avec Faster-Whisper (Production)

```python
def transcribe_audio_faster_whisper(audio_path: str) -> str:
    segments, info = model.transcribe(
        audio_path,
        language="fr",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        chunk_length=30,
        condition_on_previous_text=True,
    )
    
    # Concaténer segments
    transcription = " ".join([segment.text for segment in segments])
    return transcription
```

### 3.3 Transcription Long-Form (Réunions de 30-120 min)

```python
def transcribe_long_meeting(audio_path: str, chunk_length_s: int = 30) -> str:
    """
    Transcrit une réunion longue en segments avec overlap.
    """
    import librosa
    
    audio_array, sample_rate = librosa.load(audio_path, sr=16000)
    duration = len(audio_array) / sample_rate
    
    transcriptions = []
    overlap_s = 1.0  # 1 seconde d'overlap entre segments
    
    current_time = 0.0
    
    while current_time < duration:
        # Calculer indices
        start_idx = int(current_time * sample_rate)
        end_idx = int((current_time + chunk_length_s) * sample_rate)
        
        if end_idx > len(audio_array):
            end_idx = len(audio_array)
        
        # Extraire chunk
        chunk = audio_array[start_idx:end_idx]
        
        # Transcrire chunk
        inputs = processor(
            audio=chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_features"],
                max_length=448,
                num_beams=5,
                language="fr",
                task="transcribe",
            )
        
        chunk_transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        
        transcriptions.append(chunk_transcription)
        
        # Avancer (avec overlap)
        current_time += chunk_length_s - overlap_s
    
    # Concaténer et nettoyer (retirer répétitions potentielles)
    full_transcription = " ".join(transcriptions)
    
    # Post-processing: retirer répétitions de début/fin de chunks
    # (simplifié ici, à améliorer selon besoins)
    full_transcription = clean_overlap_repetitions(full_transcription)
    
    return full_transcription
```

## 4. Paramètres Recommandés

### 4.1 Paramètres Optimaux pour Réunions

```python
GENERATION_CONFIG = {
    "max_length": 448,           # Longueur max Whisper
    "num_beams": 5,              # Beam search (bon compromis perf/vitesse)
    "best_of": 5,                # Nombre de candidats à générer
    "temperature": 0.0,          # Déterministe (pas de sampling)
    "language": "fr",            # Forcer français
    "task": "transcribe",        # Transcription (pas traduction)
    "condition_on_previous_text": True,  # Contexte précédent
    "compression_ratio_threshold": 2.4,  # Détecte répétitions/hallucinations
    "log_prob_threshold": -1.0,  # Filtre faible confiance
    "no_speech_threshold": 0.6,  # Détecte silence
}
```

### 4.2 Paramètres selon Cas d'Usage

**Production GPU (qualité optimale) :**
```python
{
    "num_beams": 5,
    "chunk_length": 30,
    "compute_type": "float16",  # faster-whisper
}
```

**Production CPU (latence acceptable) :**
```python
{
    "num_beams": 3,  # Réduire pour vitesse
    "chunk_length": 30,
    "compute_type": "int8",  # faster-whisper ou quantifié
}
```

**Edge/Mobile (ultra frugal) :**
```python
{
    "num_beams": 1,  # Greedy (plus rapide)
    "chunk_length": 20,  # Chunks plus petits
    "compute_type": "int8",
}
```

## 5. Post-Processing et Optimisations

### 5.1 Correction Lexique (Noms Propres, Acronymes)

```python
class LexiconCorrector:
    """Corrige les erreurs fréquentes via lexique."""
    
    def __init__(self, lexicon_path: str):
        # Lexique: {"mot_mal_transcrit": "mot_correct"}
        self.lexicon = self._load_lexicon(lexicon_path)
    
    def correct(self, text: str) -> str:
        """Applique corrections du lexique."""
        words = text.split()
        corrected = []
        
        for word in words:
            # Chercher dans lexique (tolérance faute de frappe)
            if word.lower() in self.lexicon:
                corrected.append(self.lexicon[word.lower()])
            else:
                corrected.append(word)
        
        return " ".join(corrected)

# Usage
corrector = LexiconCorrector("lexicon/meetings_entities.json")
transcription = transcribe_audio("meeting.wav")
transcription_corrected = corrector.correct(transcription)
```

### 5.2 Détection et Correction Hallucinations

```python
def detect_hallucinations(text: str, threshold: float = 2.4) -> bool:
    """
    Détecte répétitions excessives (signe d'hallucination).
    Utilise compression ratio comme proxy.
    """
    words = text.split()
    unique_words = set(words)
    
    if len(unique_words) == 0:
        return True
    
    compression_ratio = len(words) / len(unique_words)
    return compression_ratio > threshold

def clean_repetitions(text: str) -> str:
    """Nettoie répétitions de mots/phrases."""
    # Implémentation simplifiée
    # À améliorer selon besoins spécifiques
    words = text.split()
    cleaned = []
    prev_word = None
    
    for word in words:
        if word != prev_word:
            cleaned.append(word)
        prev_word = word
    
    return " ".join(cleaned)
```

## 6. Intégration dans Pipeline Compte-Rendu

### 6.1 Pipeline Complet

```python
class MeetingTranscriptionPipeline:
    """Pipeline complet pour transcription de réunions."""
    
    def __init__(self, model_path: str, use_faster_whisper: bool = True):
        if use_faster_whisper:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_path, device="cuda", compute_type="float16")
            self.processor = None
        else:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self.processor = WhisperProcessor.from_pretrained(model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        self.use_faster_whisper = use_faster_whisper
    
    def process_meeting(
        self,
        audio_path: str,
        participants: list = None,
        entities: dict = None,
    ) -> dict:
        """
        Traite une réunion complète.
        
        Returns:
            Dict avec transcription, métadonnées, etc.
        """
        # Transcription
        if self.use_faster_whisper:
            transcription = self._transcribe_faster_whisper(audio_path)
        else:
            transcription = self._transcribe_transformers(audio_path)
        
        # Post-processing
        if entities:
            transcription = self._apply_lexicon_corrections(transcription, entities)
        
        transcription = self._clean_text(transcription)
        
        # Métadonnées
        metadata = {
            "duration": self._get_audio_duration(audio_path),
            "word_count": len(transcription.split()),
            "has_hallucinations": detect_hallucinations(transcription),
        }
        
        return {
            "transcription": transcription,
            "metadata": metadata,
        }
    
    def _transcribe_faster_whisper(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(
            audio_path,
            language="fr",
            beam_size=5,
            chunk_length=30,
        )
        return " ".join([s.text for s in segments])
    
    def _transcribe_transformers(self, audio_path: str) -> str:
        # Implémentation avec transformers (voir section 3.1)
        pass
    
    def _apply_lexicon_corrections(self, text: str, entities: dict) -> str:
        # Corriger via lexique
        pass
    
    def _clean_text(self, text: str) -> str:
        # Nettoyage général
        return clean_repetitions(text)
    
    def _get_audio_duration(self, audio_path: str) -> float:
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        return len(y) / sr

# Usage
pipeline = MeetingTranscriptionPipeline("outputs/models/whisper-meetings/final")
result = pipeline.process_meeting(
    "meeting_2024_01_15.wav",
    participants=["Jean Dupont", "Marie Martin"],
    entities={"acronyms": {"API": "API", "CR": "CR"}},
)
```

## 7. Déploiement

### 7.1 API REST (Flask/FastAPI)

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()
pipeline = MeetingTranscriptionPipeline("outputs/models/whisper-meetings/final")

@app.post("/transcribe")
async def transcribe_meeting(audio: UploadFile = File(...)):
    # Sauvegarder temporairement
    temp_path = f"/tmp/{audio.filename}"
    with open(temp_path, "wb") as f:
        f.write(await audio.read())
    
    # Transcrire
    result = pipeline.process_meeting(temp_path)
    
    return JSONResponse(result)
```

### 7.2 Batch Processing

```python
def process_meetings_batch(audio_files: list, output_dir: str):
    """Traite plusieurs réunions en batch."""
    import json
    
    results = []
    
    for audio_path in audio_files:
        result = pipeline.process_meeting(audio_path)
        results.append({
            "audio": audio_path,
            **result,
        })
        
        # Sauvegarder individuellement
        output_file = Path(output_dir) / f"{Path(audio_path).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    return results
```

## 8. Monitoring et Qualité

### 8.1 Métriques de Qualité

```python
def compute_quality_metrics(transcription: str) -> dict:
    """Calcule métriques de qualité de transcription."""
    return {
        "word_count": len(transcription.split()),
        "avg_word_length": np.mean([len(w) for w in transcription.split()]),
        "has_repetitions": detect_hallucinations(transcription),
        "confidence_score": None,  # Si disponible depuis modèle
    }
```

### 8.2 Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribe_with_logging(audio_path: str):
    logger.info(f"Début transcription: {audio_path}")
    
    start_time = time.time()
    result = pipeline.process_meeting(audio_path)
    duration = time.time() - start_time
    
    logger.info(f"Transcription terminée en {duration:.2f}s")
    logger.info(f"Qualité: {compute_quality_metrics(result['transcription'])}")
    
    return result
```

## 9. Troubleshooting

### Problèmes Courants

**1. Erreur "CUDA out of memory"**
- Réduire `batch_size` ou `chunk_length`
- Utiliser `float16` au lieu de `float32`
- Utiliser quantization int8

**2. Latence trop élevée**
- Utiliser `faster-whisper` au lieu de `transformers`
- Réduire `num_beams` (3 ou 1)
- Utiliser modèle quantifié

**3. Qualité insuffisante sur noms propres**
- Ajouter lexique de correction
- Ré-entraîner avec sur-échantillonnage de segments contenant noms propres
- Utiliser LoRA spécialisé vocabulaire

**4. Hallucinations sur longues réunions**
- Réduire `chunk_length` (20s au lieu de 30s)
- Augmenter `compression_ratio_threshold`
- Post-processing pour détecter répétitions

## 10. Limitations et Next Steps

### Limitations Actuelles

- Pas de diarisation intégrée (speaker identification)
- Traitement séquentiel (pas de streaming réel)
- Pas d'adaptation automatique par secteur

### Améliorations Futures

- Intégration diarisation (pyannote.audio)
- Streaming transcription (<2s latence)
- Adaptation continue avec feedback utilisateur
- Support multilingue (français + anglais)

