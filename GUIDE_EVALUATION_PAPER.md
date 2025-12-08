# Guide d'évaluation complète pour publication

Ce guide explique comment générer toutes les métriques nécessaires pour votre papier.

## 📊 Métriques mesurées

1. **Performance (qualité)**:
   - WER (Word Error Rate) sur différents datasets
   - CER (Character Error Rate)
   - Métriques par dataset (Common Voice, MLS, etc.)

2. **Performance (vitesse)**:
   - Latence par échantillon
   - Débit (throughput)
   - Mémoire VRAM utilisée

3. **Frugalité**:
   - Taille du modèle (GB)
   - Nombre de paramètres
   - Mémoire RAM/VRAM

4. **Comparaison**:
   - vs Whisper Large-v3 (baseline)
   - Accélération (speedup)
   - Réduction de taille
   - Dégradation qualité

## 🚀 Étapes d'évaluation

### 1. Évaluation complète (modèle + baseline)

```bash
cd /workspace/finetuning_gilbert
git pull

# Évaluation complète (prend 10-30 minutes selon datasets)
python scripts/evaluate_comprehensive.py \
    --model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
    --baseline_model openai/whisper-large-v3 \
    --device cuda \
    --max_samples 100 \
    --output outputs/evaluations/comprehensive_results.json
```

**Options importantes**:
- `--max_samples`: Nombre d'échantillons par dataset (défaut: 100)
- `--skip_baseline`: Pour skip l'évaluation baseline (plus rapide)
- `--device`: `cuda` ou `cpu`

### 2. Générer les tableaux pour le papier

```bash
# Générer tableaux LaTeX et Markdown
python scripts/generate_publication_table.py \
    --results outputs/evaluations/comprehensive_results.json \
    --format both \
    --output outputs/evaluations/publication_table.md
```

### 3. Benchmark vitesse détaillé

```bash
# Benchmark vitesse avec plus de runs pour statistiques robustes
python scripts/benchmark_model.py \
    --model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
    --device cuda \
    --num_runs 20  # Plus de runs = statistiques plus robustes
```

## 📋 Structure des résultats

Le fichier JSON contient:

```json
{
  "model_name": "...",
  "model_size_gb": 1.51,
  "num_parameters": 750000000,
  "inference_benchmark": {
    "mean_time": 0.03,
    "std_time": 0.001,
    "peak_memory_gb": 1.57
  },
  "quality_metrics": {
    "common_voice_fr": {
      "wer": 0.05,
      "cer": 0.02,
      "num_samples": 100
    }
  },
  "average_wer": 0.05,
  "average_cer": 0.02,
  "speedup_vs_baseline": 4.2,
  "size_reduction_percent": 50.0,
  "wer_degradation_vs_baseline": 0.01
}
```

## 📄 Utilisation pour le papier

### Métriques principales à mentionner

1. **Efficacité**:
   - Taille: X GB (réduction de Y% vs baseline)
   - Accélération: Xx plus rapide
   - Mémoire: X GB VRAM

2. **Qualité**:
   - WER moyen: X%
   - CER moyen: X%
   - Dégradation: +X% vs baseline (si applicable)

3. **Comparaison**:
   - Tableau comparatif automatiquement généré
   - Graphiques possibles avec les données JSON

### Exemple de section pour papier

```
Le modèle Whisper-Large-V3-Distil-French-v0.2 a été évalué sur 
[datasets] et comparé à Whisper Large-v3. Les résultats montrent:

- Taille: 1.51 GB (réduction de 50% vs baseline)
- Vitesse: 4.2x plus rapide
- Qualité: WER de 5.2% (dégradation de 0.8% vs baseline)
- Mémoire: 1.57 GB VRAM

Ces résultats démontrent un excellent compromis qualité/frugalité...
```

## 🔬 Datasets recommandés pour publication

Pour un papier robuste, évaluer sur:

1. **Common Voice French** (standard, généraliste)
2. **MLS French** (haute qualité, lecture)
3. **VoxPopuli French** (parlementaire, proche réunions)
4. **Dataset interne** (si disponible, spécifique réunions)

## ⚠️ Notes importantes

- Les résultats varient selon les datasets
- Plus d'échantillons = statistiques plus robustes (mais plus long)
- Le baseline (large-v3) est plus lourd à évaluer, utilisez `--skip_baseline` pour tests rapides
- Les métriques VRAM dépendent du GPU utilisé

## 📊 Visualisations (optionnel)

Les données JSON peuvent être utilisées pour créer des graphiques:

```python
import json
import matplotlib.pyplot as plt

with open("outputs/evaluations/comprehensive_results.json") as f:
    results = json.load(f)

# Créer graphiques comparatifs, etc.
```

## 🎯 Métriques spécifiques réunions (si données disponibles)

Si vous avez un dataset de réunions:

```python
# Utiliser le script avec votre dataset custom
python scripts/evaluate_comprehensive.py \
    --model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
    --custom_dataset data/test_sets/meetings_test.json \
    --output outputs/evaluations/meetings_evaluation.json
```

## ✅ Checklist pour papier

- [ ] Évaluation sur au moins 2 datasets publics
- [ ] Comparaison avec baseline (large-v3)
- [ ] Métriques de vitesse (latence, throughput)
- [ ] Métriques de qualité (WER, CER)
- [ ] Métriques de frugalité (taille, mémoire)
- [ ] Tableau comparatif généré
- [ ] Statistiques robustes (suffisamment d'échantillons)
- [ ] Métriques sur données de réunions (si disponible)

