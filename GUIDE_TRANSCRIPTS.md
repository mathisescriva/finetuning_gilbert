# Guide : Génération Automatique de Transcripts

## 🎯 Objectif

Ce guide explique comment générer automatiquement des transcripts pour un dataset audio qui n'en a pas. Deux approches sont disponibles :

1. **Services commerciaux** (AssemblyAI, Deepgram, etc.) - ⭐ **RECOMMANDÉ** pour meilleure qualité
2. **Whisper gratuit** - Alternative économique mais qualité moindre

## 💡 Pourquoi utiliser un service commercial ?

**Avantages :**
- ✅ **Meilleure qualité** : Services optimisés pour production
- ✅ **Plus rapide** : API optimisées, traitement parallèle
- ✅ **Fonctionnalités avancées** : Diarisation, ponctuation, timestamps
- ✅ **Moins d'erreurs** : Meilleure reconnaissance noms propres, accents

**Inconvénients :**
- ⚠️ **Coût** : ~$0.0001-0.001 par minute audio
- ⚠️ **Dépendance API** : Nécessite connexion internet

**Recommandation** : Pour des pseudo-labels de qualité maximale (et donc meilleur fine-tuning), utilisez un service commercial si le budget le permet.

## 🚀 Utilisation Rapide

### Option 1 : Service Commercial (RECOMMANDÉ) ⭐

**AssemblyAI** (meilleur rapport qualité/prix) :

```bash
# 1. Obtenir une clé API gratuite (50$ de crédit) : https://www.assemblyai.com
# 2. Définir la clé
export ASSEMBLYAI_API_KEY="votre_cle_api"

# 3. Générer transcripts
python scripts/generate_transcripts_commercial.py \
  --dataset_name MEscriva/french-education-speech \
  --service assemblyai \
  --split train
```

**Deepgram** (alternative performante) :

```bash
export DEEPGRAM_API_KEY="votre_cle_api"
python scripts/generate_transcripts_commercial.py \
  --dataset_name MEscriva/french-education-speech \
  --service deepgram
```

### Option 2 : Whisper Gratuit (Alternative)

```bash
# Génération avec Whisper (gratuit mais qualité moindre)
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech \
  --split train \
  --output_dir data/processed

# Ou avec Makefile
make generate-transcripts
```

## 📋 Options Détaillées

### Commandes de Base

```bash
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech \
  --split train \
  --output_dir data/processed \
  --output_name french_education_with_transcripts
```

### Options Utiles

**Test rapide (limiter le nombre d'échantillons) :**
```bash
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech \
  --max_samples 10  # Test avec 10 échantillons seulement
```

**Filtrer par confidence (garder seulement transcripts fiables) :**
```bash
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech \
  --min_confidence 0.7  # Garder seulement confidence >= 0.7
```

**Utiliser un autre modèle Whisper :**
```bash
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech \
  --model_name openai/whisper-large-v3  # Modèle plus puissant mais plus lent
```

**Pousser le dataset sur HuggingFace Hub :**
```bash
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech \
  --push_to_hub \
  --hub_token YOUR_TOKEN
```

## 🔍 Fonctionnement

### Processus

1. **Chargement du dataset** : Le script charge votre dataset depuis HuggingFace
2. **Chargement du modèle Whisper** : Utilise `bofenghuang/whisper-large-v3-distil-fr-v0.2` par défaut
3. **Transcription** : Pour chaque audio, génère un transcript automatique
4. **Calcul de confidence** : Estime la confiance de chaque transcript
5. **Sauvegarde** : Crée un nouveau dataset avec les transcripts ajoutés

### Format de Sortie

Le dataset généré contient :
- **Colonne originale `audio`** : Conservée
- **Nouvelle colonne `text`** : Transcripts générés automatiquement
- **Nouvelle colonne `transcription_confidence`** : Score de confiance (0-1)
- **Nouvelle colonne `auto_generated`** : `True` pour tous les transcripts auto

### Structure de Sortie

```
data/processed/
└── MEscriva_french-education-speech_with_transcripts/
    ├── train/
    │   ├── dataset_info.json
    │   └── state.json
    └── transcripts.json  # Export JSON pour référence
```

## 📊 Statistiques et Qualité

### Interprétation des Scores de Confiance

- **0.8 - 1.0** : Très fiable ✅
- **0.6 - 0.8** : Fiable ✅
- **0.4 - 0.6** : À vérifier ⚠️
- **< 0.4** : Faible, probablement erreur ❌

### Filtrage Recommandé

Pour un fine-tuning de qualité, filtrez les transcripts de faible confidence :

```bash
# Garder seulement transcripts confiants
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech \
  --min_confidence 0.6
```

## ✅ Vérification et Correction Manuelle (Optionnel)

### Exporter les transcripts pour vérification

Le script génère aussi un fichier `transcripts.json` :

```json
[
  {
    "index": 0,
    "text": "Bonjour, bienvenue dans ce cours...",
    "confidence": 0.85
  },
  {
    "index": 1,
    "text": "...",
    "confidence": 0.45
  }
]
```

### Identifier les transcripts à vérifier

```python
import json

with open("data/processed/.../transcripts.json", 'r') as f:
    transcripts = json.load(f)

# Trouver les transcripts de faible confidence
low_confidence = [t for t in transcripts if t["confidence"] < 0.6]
print(f"Transcripts à vérifier: {len(low_confidence)}")
```

## 🔄 Utilisation pour Fine-tuning

Une fois les transcripts générés, utilisez directement le dataset pour fine-tuning :

```bash
python scripts/fine_tune_meetings.py \
  --train_data data/processed/MEscriva_french-education-speech_with_transcripts \
  --eval_data data/processed/MEscriva_french-education-speech_with_transcripts \
  --phase 1
```

## 💡 Stratégies d'Amélioration

### 1. Améliorer la Qualité des Transcripts

**Option A : Utiliser un modèle plus puissant**
```bash
--model_name openai/whisper-large-v3  # Plus lent mais meilleure qualité
```

**Option B : Post-processing**
- Corriger les erreurs fréquentes manuellement
- Utiliser un lexique de correction (noms propres, termes spécialisés)

### 2. Pseudo-labeling Itératif

1. Générer transcripts avec modèle de base
2. Fine-tuner sur ces transcripts
3. Régénérer transcripts avec modèle fine-tuné
4. Répéter jusqu'à convergence

```bash
# Étape 1 : Générer avec modèle de base
python scripts/generate_transcripts.py --dataset_name MEscriva/french-education-speech

# Étape 2 : Fine-tuner
python scripts/fine_tune_meetings.py \
  --train_data data/processed/.../with_transcripts \
  --phase 1

# Étape 3 : Régénérer avec modèle fine-tuné
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech \
  --model_name outputs/models/whisper-meetings-phase1/final
```

### 3. Combinaison avec Données Manuelles

- Générer transcripts automatiques pour la majorité des données
- Annoter manuellement un sous-ensemble (10-20%) pour validation/qualité
- Mélanger les deux pour fine-tuning

## ⚙️ Paramètres Avancés

### Performance

**GPU recommandé** : La transcription est beaucoup plus rapide sur GPU.

```bash
# Forcer CPU (plus lent)
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech \
  --device cpu
```

**Traitement par batch** : Le script traite un échantillon à la fois (car durées audio variables), mais sauvegarde périodiquement pour éviter la perte en cas d'interruption.

### Gestion Mémoire

Pour datasets très volumineux :
- Utiliser `--max_samples` pour traiter par chunks
- Traiter séparément train/validation/test
- Sauvegardes intermédiaires automatiques tous les 100 échantillons

## 🐛 Troubleshooting

### Erreur "Colonne audio non trouvée"

Le script essaie automatiquement `audio`, `path`, `file`. Si votre dataset utilise un autre nom :
- Vérifiez les colonnes : `dataset.column_names`
- Modifiez le script si nécessaire (section identification colonne audio)

### Erreur "CUDA out of memory"

- Réduire `batch_size` (actuellement 1, donc peu probable)
- Utiliser `--device cpu`
- Traiter par chunks avec `--max_samples`

### Transcripts de mauvaise qualité

- Vérifier qualité audio (bruit, débit de parole)
- Utiliser modèle plus puissant (`whisper-large-v3`)
- Filtrer par `--min_confidence` plus élevé
- Post-processing manuel des erreurs fréquentes

## 📝 Exemple Complet

```bash
# 1. Générer transcripts (test avec 50 échantillons)
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech \
  --split train \
  --max_samples 50 \
  --output_dir data/processed

# 2. Vérifier les résultats
cat data/processed/MEscriva_french-education-speech_with_transcripts/transcripts.json | head -20

# 3. Si satisfait, générer pour tout le dataset
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech \
  --split train \
  --min_confidence 0.6 \
  --output_dir data/processed

# 4. Fine-tuning
python scripts/fine_tune_meetings.py \
  --train_data data/processed/MEscriva_french-education-speech_with_transcripts \
  --eval_data data/processed/MEscriva_french-education-speech_with_transcripts \
  --phase 1
```

## 🎓 Bonnes Pratiques

1. **Toujours tester d'abord** avec `--max_samples 10-50`
2. **Vérifier la qualité** des transcripts générés avant de tout traiter
3. **Filtrer par confidence** pour éviter de polluer l'entraînement
4. **Sauvegarder périodiquement** (fait automatiquement)
5. **Itérer** : générer → fine-tuner → régénérer pour amélioration

---

**Note** : Les transcripts générés sont des "pseudo-labels" - ils ne sont pas parfaits mais constituent une excellente base pour le fine-tuning, surtout si combinés avec quelques données manuellement annotées.

