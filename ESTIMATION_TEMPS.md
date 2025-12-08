# Estimation Temps d'Entraînement QAT

## ⏱️ Temps Réel (Corrigé)

### Sur GPU Moderne (A100/V100/RTX 3090)

**Configuration Optimisée (Par Défaut)** :
- Dataset : 60,000 samples (~500h de segments 30s)
- Époques : 5
- Batch size : 8
- Gradient accumulation : 4
- **Temps total : 2-4 heures** ⚡

**Configuration Extended** :
- Dataset : 120,000 samples (~1000h)
- Époques : 10
- Batch size : 8
- **Temps total : 6-10 heures**

### Sur CPU

- **Temps total : 1-2 jours** (beaucoup plus lent, pas recommandé)

## 📊 Détails du Calcul

### Pourquoi c'est si rapide ?

1. **Modèle pré-entraîné** : On part de v0.2, pas depuis zéro
2. **Moins d'époques** : 5 vs 160 pour entraînement initial
3. **Sous-ensemble dataset** : 500h vs 10,000h pour entraînement initial
4. **Modèle distillé rapide** : ~0.1x RTF en inference, ~0.3-0.5x RTF en training

### Calcul Détaillé

```
Dataset: 60,000 segments × 30s = 500h audio
Effective batch: 8 × 4 = 32
Steps par epoch: 60,000 / 32 = 1,875 steps
Temps par step: ~0.5-1s (GPU moderne)
Temps par epoch: 1,875 × 0.75s = ~23 minutes
Total (5 epochs): ~2 heures
```

*(+ overhead I/O, validation, etc. = 2-4h total)*

## 🎯 Recommandation

**Utilisez les paramètres par défaut** (optimisés) :
- ✅ 2-4h de training (rapide)
- ✅ Qualité excellente (suffisant pour publication)
- ✅ Peut toujours étendre ensuite si besoin

## 🔧 Ajustements Possibles

Si vous avez plus de temps GPU disponible :

```bash
# Version extended (6-10h) - Qualité maximale
python scripts/train_qat.py \
  --num_epochs 10 \
  --max_samples 120000
```

Si GPU limité :

```bash
# Version rapide (1-2h) - Minimum viable
python scripts/train_qat.py \
  --num_epochs 3 \
  --max_samples 30000 \
  --per_device_batch_size 4
```

