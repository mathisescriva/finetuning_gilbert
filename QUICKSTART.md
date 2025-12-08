# Quickstart : Fine-tuning Whisper pour Réunions

## Démarrage Rapide

### 1. Installation

```bash
# Cloner ou naviguer dans le projet
cd finetuning_gilbert

# Installer dépendances
pip install -r requirements.txt
```

### 2. Télécharger des Données Publiques 🇫🇷

**Option A : Télécharger automatiquement Common Voice français (RECOMMANDÉ)**

```bash
# Télécharger Common Voice français (dataset public gratuit)
python scripts/download_datasets.py --datasets common_voice

# Ou avec le Makefile
make download-datasets
```

Cela télécharge ~100+ heures de données françaises dans `data/processed/common_voice_fr/`.

**Option B : Utiliser vos propres données de réunions**

Si vous avez vos propres données, créer un fichier JSON :

```json
[
  {
    "audio": "path/to/meeting1.wav",
    "text": "Transcription de référence de la réunion..."
  },
  {
    "audio": "path/to/meeting2.wav",
    "text": "Autre transcription..."
  }
]
```

Sauvegarder dans `data/raw/train_data.json` et `data/raw/eval_data.json`.

**Voir `DATASETS.md` pour plus d'options de datasets publics.**

### 3. Évaluation Baseline

Évaluer le modèle de base pour établir la baseline :

```bash
# Si vous avez téléchargé Common Voice
python scripts/evaluate_baseline.py \
  --model_name bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --test_data data/processed/common_voice_fr \
  --output_dir outputs/evaluations

# Ou avec vos données JSON
python scripts/evaluate_baseline.py \
  --model_name bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --test_data data/raw/eval_data.json \
  --output_dir outputs/evaluations
```

### 4. Fine-tuning

#### Phase 1 : Encoder Frozen

**Avec dataset HuggingFace (Common Voice)** :
```bash
python scripts/fine_tune_meetings.py \
  --model_name bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --train_data data/processed/common_voice_fr \
  --eval_data data/processed/common_voice_fr \
  --output_dir outputs/models/whisper-meetings-phase1 \
  --phase 1
```

**Avec vos données JSON** :
```bash
python scripts/fine_tune_meetings.py \
  --model_name bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --train_data data/raw/train_data.json \
  --eval_data data/raw/eval_data.json \
  --output_dir outputs/models/whisper-meetings-phase1 \
  --phase 1
```

#### Phase 2 : Full Fine-tuning

```bash
python scripts/fine_tune_meetings.py \
  --model_name outputs/models/whisper-meetings-phase1/final \
  --train_data data/raw/train_data.json \
  --eval_data data/raw/eval_data.json \
  --output_dir outputs/models/whisper-meetings-phase2 \
  --phase 2
```

#### Phase 3 : LoRA (Optionnel)

```bash
python scripts/fine_tune_meetings.py \
  --model_name bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --train_data data/raw/train_data.json \
  --eval_data data/raw/eval_data.json \
  --output_dir outputs/models/whisper-meetings-lora \
  --phase 3 \
  --use_lora
```

### 5. Quantization (Optionnel)

Quantifier le modèle en int8 pour réduire taille et latence :

```bash
python scripts/distill_quantize.py \
  --model_path outputs/models/whisper-meetings-phase2/final \
  --output_dir outputs/models/whisper-meetings-int8 \
  --quantization_type int8
```

### 6. Benchmark Comparatif

Comparer différents modèles :

```bash
python scripts/benchmark.py \
  --test_data data/raw/eval_data.json \
  --models \
    openai/whisper-large-v3 \
    bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --custom_models \
    outputs/models/whisper-meetings-phase2/final \
  --output_dir outputs/evaluations
```

### 7. Utilisation du Modèle

Voir `GUIDE_INTEGRATION.md` pour les détails complets.

**Exemple rapide avec Transformers :**

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Charger modèle
model_name = "outputs/models/whisper-meetings-phase2/final"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Transcrire
audio, sr = librosa.load("meeting.wav", sr=16000)
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(inputs["input_features"], language="fr")
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)
```

## Structure des Données

### Format JSON Attendu

```json
{
  "test_samples": [
    {
      "audio": "path/to/audio.wav",
      "text": "Transcription complète..."
    }
  ],
  "entities": ["Nom1", "Nom2", "ACRONYME"],  // Optionnel
  "metadata": {
    "description": "...",
    "language": "fr"
  }
}
```

### Formats Audio Supportés

- WAV (recommandé)
- MP3
- FLAC
- OGG

Sample rate sera automatiquement converti à 16 kHz.

## Configuration

Modifier `config/training_config.yaml` et `config/model_config.yaml` selon besoins.

### Paramètres Clés

**Training :**
- `learning_rate` : Taux d'apprentissage (défaut: 1e-5)
- `per_device_train_batch_size` : Taille batch (défaut: 8)
- `num_epochs` : Nombre d'époques (défaut: 3-5)

**Inference :**
- `beam_size` : Taille beam search (défaut: 5)
- `chunk_length_s` : Longueur chunks (défaut: 30)

## Troubleshooting

**Erreur CUDA OOM :**
- Réduire `per_device_train_batch_size`
- Augmenter `gradient_accumulation_steps`
- Utiliser `fp16: true` dans config

**Qualité insuffisante :**
- Vérifier qualité données (transcriptions précises)
- Augmenter nombre d'époques
- Ajouter plus de données d'entraînement
- Vérifier augmentations audio (peuvent être trop agressives)

**Latence élevée :**
- Utiliser `faster-whisper` au lieu de `transformers`
- Réduire `beam_size` (3 ou 1)
- Quantifier modèle (int8)

## Ressources

- **Plan technique** : `PLAN_TECHNIQUE.md`
- **Guide intégration** : `GUIDE_INTEGRATION.md`
- **Limites & next steps** : `LIMITES_ET_NEXT_STEPS.md`

