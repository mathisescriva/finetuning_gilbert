# Fine-tuning Whisper pour Comptes-Rendus de Réunion

## Vue d'ensemble

Ce projet vise à transformer le modèle `bofenghuang/whisper-large-v3-distil-fr-v0.2` en un modèle spécialisé pour la transcription de réunions en français, en optimisant le ratio performance/frugalité.

## Structure du projet

```
.
├── README.md                          # Ce fichier
├── PLAN_TECHNIQUE.md                  # Plan technique détaillé
├── requirements.txt                   # Dépendances Python
├── config/                            # Configurations
│   ├── model_config.yaml             # Configuration du modèle
│   └── training_config.yaml          # Configuration d'entraînement
├── scripts/
│   ├── download_datasets.py          # Télécharger datasets publics français
│   ├── generate_transcripts.py       # Générer transcripts automatiques (pseudo-labels)
│   ├── evaluate_baseline.py          # Évaluation du modèle de base
│   ├── fine_tune_meetings.py         # Fine-tuning sur réunions
│   ├── distill_quantize.py           # Distillation et quantization
│   └── benchmark.py                  # Benchmark complet
├── notebooks/
│   ├── 01_model_analysis.ipynb       # Analyse du modèle de base
│   ├── 02_data_exploration.ipynb     # Exploration des données
│   └── 03_evaluation_results.ipynb   # Visualisation des résultats
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── whisper_lora.py           # Architecture LoRA pour Whisper
│   │   └── quantized_whisper.py      # Modèle quantifié
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # Dataset pour réunions
│   │   └── augmentations.py          # Augmentations audio
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Trainer personnalisé
│   │   └── distillation.py           # Distillation
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                # Métriques WER/CER
│       └── evaluator.py              # Évaluateur complet
├── data/
│   ├── raw/                          # Données brutes (non versionnées)
│   ├── processed/                    # Données traitées
│   └── test_sets/                    # Jeux de test
└── outputs/
    ├── models/                       # Modèles entraînés
    ├── logs/                         # Logs d'entraînement
    └── evaluations/                  # Résultats d'évaluation

```

## Installation

```bash
pip install -r requirements.txt
```

## Démarrage Rapide

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Préparer les Données

**Option A : Télécharger datasets publics**
```bash
python scripts/download_datasets.py --datasets common_voice
make download-datasets
```

**Option B : Générer transcripts automatiques pour votre dataset audio**
Si vous avez un dataset audio sans transcripts (ex: `MEscriva/french-education-speech`) :
```bash
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech \
  --split train
make generate-transcripts
```
Voir `GUIDE_TRANSCRIPTS.md` pour plus de détails.

### 3. Fine-tuning

```bash
python scripts/fine_tune_meetings.py \
  --train_data data/processed/common_voice_fr \
  --eval_data data/processed/common_voice_fr \
  --phase 1
```

## Documentation Complète

- **QUICK_START_QAT.md** : 🚀 Lancer QAT depuis CLI (recommandé)
- **QUICKSTART.md** : Guide de démarrage rapide général
- **DATASETS.md** : Guide des datasets disponibles
- **GUIDE_TRANSCRIPTS.md** : Génération automatique de transcripts (Whisper)
- **GUIDE_QAT.md** : Guide complet QAT (Quantization-Aware Training)
- **SERVICES_COMPARAISON.md** : Comparaison services commerciaux (AssemblyAI, Deepgram, etc.)
- **PLAN_TECHNIQUE.md** : Plan technique détaillé
- **GUIDE_INTEGRATION.md** : Guide d'intégration du modèle
- **LIMITES_ET_NEXT_STEPS.md** : Limitations et améliorations futures

## Licence

MIT (héritée du modèle de base)

