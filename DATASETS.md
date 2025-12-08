# Guide des Datasets pour Fine-tuning

## 📥 Téléchargement Automatique

Le script `scripts/download_datasets.py` télécharge automatiquement des datasets publics français disponibles sur HuggingFace.

### Utilisation

```bash
# Télécharger Common Voice français (recommandé pour commencer)
python scripts/download_datasets.py --datasets common_voice

# Télécharger tous les datasets disponibles
python scripts/download_datasets.py --datasets all --combine

# Limiter à 1000 échantillons pour test rapide
python scripts/download_datasets.py --datasets common_voice --max_samples 1000
```

## 📚 Datasets Disponibles

### 1. Mozilla Common Voice Français ⭐ RECOMMANDÉ

**Qualité :** ⭐⭐⭐⭐  
**Taille :** ~100+ heures  
**Style :** Parole naturelle, variée  
**Usage :** Excellent pour fine-tuning général

```bash
python scripts/download_datasets.py --datasets common_voice
```

**Avantages :**
- Large volume de données
- Qualité vérifiée par communauté
- Diversité d'accents et de voix
- Format directement compatible HuggingFace

**Inconvénients :**
- Pas spécifiquement des réunions (parole générale)
- Peut nécessiter filtrage pour données de meilleure qualité

### 2. Multilingual LibriSpeech (MLS) Français

**Qualité :** ⭐⭐⭐⭐⭐  
**Taille :** ~500+ heures  
**Style :** Lecture de livres (plus formel)  
**Usage :** Complément à Common Voice

```bash
python scripts/download_datasets.py --datasets mls
```

**Avantages :**
- Très haute qualité audio
- Prononciation claire
- Vocabulaire riche

**Inconvénients :**
- Style plus formel (lecture vs conversation)
- Moins proche du style réunions

### 3. VoxPopuli Français

**Qualité :** ⭐⭐⭐⭐  
**Taille :** Très grande  
**Style :** Données parlementaires européennes  
**Usage :** Plus proche du style réunions formelles

```bash
python scripts/download_datasets.py --datasets voxpopuli
```

**Avantages :**
- Style conversationnel/parlé
- Proche du contexte réunions (discussions formelles)
- Qualité audio généralement bonne

**Inconvénients :**
- Peut contenir du vocabulaire spécialisé politique
- Format peut varier selon la source

## 🔄 Combiner Plusieurs Datasets

Pour créer un dataset mixte plus riche :

```bash
python scripts/download_datasets.py \
  --datasets common_voice mls voxpopuli \
  --combine \
  --max_samples 5000
```

Cela crée un dataset combiné dans `data/processed/combined_french_asr/`.

## 💡 Stratégie Recommandée

### Pour Démarrage Rapide

1. **Common Voice uniquement** :
   ```bash
   python scripts/download_datasets.py --datasets common_voice
   ```

2. **Utiliser directement avec fine-tuning** :
   ```bash
   python scripts/fine_tune_meetings.py \
     --train_data data/processed/common_voice_fr \
     --eval_data data/processed/common_voice_fr \
     --phase 1
   ```

### Pour Qualité Maximale

1. **Combiner Common Voice + MLS** :
   ```bash
   python scripts/download_datasets.py \
     --datasets common_voice mls \
     --combine
   ```

2. **Fine-tuning avec dataset combiné** :
   ```bash
   python scripts/fine_tune_meetings.py \
     --train_data data/processed/combined_french_asr \
     --eval_data data/processed/combined_french_asr \
     --phase 1
   ```

### Pour Style Réunions

1. **Common Voice + VoxPopuli** :
   ```bash
   python scripts/download_datasets.py \
     --datasets common_voice voxpopuli \
     --combine
   ```

## 📊 Statistiques Attendues

### Common Voice FR
- Train : ~50-100k échantillons
- Validation : ~5-10k échantillons
- Test : ~5-10k échantillons
- Durée totale : ~100+ heures

### MLS French
- Train : ~100k+ échantillons
- Durée totale : ~500+ heures

### VoxPopuli FR
- Très variable selon version
- Plusieurs milliers d'heures disponibles

## ⚠️ Limitations

### Pas de Datasets Spécifiques "Réunions"

Les datasets publics spécifiques aux réunions en français sont très rares. Les options sont :

1. **Utiliser datasets généraux** (Common Voice, MLS) → bonne base
2. **Utiliser données parlementaires** (VoxPopuli) → plus proche du style
3. **Collecter vos propres données** → idéal mais nécessite annotation

### Adaptation Nécessaire

Ces datasets ne sont pas des réunions réelles, donc :
- Fine-tuning améliorera la qualité générale
- Mais spécificités réunions (noms propres, jargon) nécessiteront vos données
- Considérez comme "pre-training" puis fine-tune sur vraies réunions

## 🎯 Prochaines Étapes

1. **Télécharger Common Voice** (démarrage rapide)
2. **Fine-tuning Phase 1** sur Common Voice
3. **Collecter vraies données réunions** (même petites quantités)
4. **Fine-tuning Phase 2** sur mix Common Voice + vos réunions
5. **Évaluer** sur test set de réunions réelles

## 🔗 Ressources Additionnelles

### Datasets HuggingFace Français

Explorez sur [HuggingFace Datasets](https://huggingface.co/datasets?language=fr&task_categories=task_categories:automatic-speech-recognition) :

- `mozilla-foundation/common_voice_*` : Common Voice
- `facebook/multilingual_librispeech` : MLS
- `facebook/voxpopuli` : VoxPopuli
- `gigaspeech/s1` : Gigaspeech (si disponible FR)

### Collecte de Vos Propres Données

Si vous avez accès à des réunions :
1. Enregistrer avec consentement
2. Transcrire manuellement (ou utiliser modèle baseline)
3. Vérifier/corriger transcriptions
4. Formater en JSON (voir `data/test_sets/example_test_data.json`)

## 📝 Format de Données

Les datasets téléchargés sont au format HuggingFace DatasetDict avec :
- Colonne `audio` : objet audio (chemin + array)
- Colonne `text` : transcription

Ils sont directement compatibles avec nos scripts de fine-tuning.

