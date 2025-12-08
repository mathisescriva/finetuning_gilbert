# Guide : Quantization-Aware Training (QAT) pour Whisper

## 🎯 Objectif

Améliorer les performances d'un modèle Whisper après quantization (int8/int4) en l'entraînant avec fake quantization.

## 📊 Complexité : Modérée ✅

**Pourquoi c'est faisable :**
- ✅ PyTorch a des outils intégrés (`torch.quantization`)
- ✅ Optimum/HuggingFace fournit des helpers
- ✅ Même infrastructure que fine-tuning classique
- ✅ Pas besoin de nouveaux datasets

**Points d'attention :**
- ⚠️ Nécessite un peu de compréhension de la quantization
- ⚠️ Entraînement plus long (10+ époques vs 3-5)
- ⚠️ Int4 nécessite implémentation custom (int8 est standard)

## 🚀 Workflow Complet

### Étape 1 : Entraînement QAT (Optimisé : 2-4h sur GPU)

```bash
# Entraîner avec QAT (int8) - Version optimisée
python scripts/train_qat.py \
  --base_model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --train_data data/processed/common_voice_fr \
  --eval_data data/processed/common_voice_fr \
  --quantization_type int8 \
  --num_epochs 5 \
  --max_samples 60000 \
  --per_device_batch_size 8 \
  --output_dir outputs/models/whisper-qat-int8

# Ou avec Makefile (paramètres optimisés par défaut)
make train-qat-int8
```

**⏱️ Temps estimé : 2-4 heures sur GPU moderne (A100/V100)**

**Ce que fait le script :**
1. Charge le modèle v0.2
2. Active fake quantization (simule int8 pendant training)
3. Entraîne le modèle pour qu'il apprenne à résister à la quantization
4. Utilise sous-ensemble optimisé (60k samples ≈ 500h) pour accélérer
5. Sauvegarde le modèle préparé pour quantization

**Paramètres optimisés par défaut :**
- `num_epochs=5` : Suffisant car modèle déjà pré-entraîné
- `max_samples=60000` : ~500h de données (vs 1000h+ complet)
- `batch_size=8` : Plus grand pour GPU, accélère training
- **Temps total : 2-4h sur GPU** (vs 6-12h avec paramètres standard)

### Étape 2 : Conversion en Modèle Quantifié

```bash
# Convertir en modèle quantifié réel (ONNX)
python scripts/convert_qat_to_quantized.py \
  --model_path outputs/models/whisper-qat-int8/final \
  --output_path outputs/models/whisper-qat-int8-quantized \
  --quantization_type int8 \
  --format onnx
```

**Ce que fait le script :**
1. Charge le modèle QAT entraîné
2. Convertit en format ONNX
3. Applique la quantization réelle (int8)
4. Sauvegarde modèle prêt pour inférence

### Étape 3 : Évaluation sur les Mêmes Corpus

```bash
# Évaluer sur les corpus de la model card
python scripts/evaluate_qat.py \
  --model_path outputs/models/whisper-qat-int8-quantized \
  --baseline_model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --corpora community-v2 mtedx zaion5 zaion6 \
  --test_data data/test_sets/eval_data.json
```

**Ce que fait le script :**
1. Évalue le modèle quantifié sur les mêmes corpus que v0.2
2. Compare avec le baseline (v0.2 non quantifié)
3. Calcule la dégradation WER
4. Génère rapport de comparaison

## 📈 Résultats Attendus

### Objectifs QAT

| Métrique | Avant QAT (PTQ) | Après QAT | Objectif |
|----------|-----------------|-----------|----------|
| **WER dégradation int8** | 1-3% | <0.5% | ✅ |
| **WER dégradation int4** | 3-5% | <2% | ✅ |
| **Taille modèle** | 50% | 25% (int8) / 12.5% (int4) | ✅ |
| **Vitesse CPU** | +2x | +3-4x | ✅ |

### Comparaison avec v0.2

Le modèle QAT devrait avoir :
- ✅ Même WER que v0.2 (avant quantization)
- ✅ Dégradation <0.5% en int8 (vs 1-3% sans QAT)
- ✅ Utilisable en int4 avec dégradation acceptable (<2%)

## 📋 Corpus d'Évaluation

Vous pouvez évaluer sur **exactement les mêmes corpus** que dans la model card :

### Corpus Publics

1. **community-v2/dev_data** : Common Voice français
2. **mtedx** : MTEDx français (lectures TED Talks)
3. **zaion5** : Dataset interne Zaion Lab (call centers)
4. **zaion6** : Dataset interne Zaion Lab (call centers)

### Utilisation

```bash
# Évaluer sur corpus publics
python scripts/evaluate_qat.py \
  --model_path outputs/models/whisper-qat-int8-quantized \
  --corpora community-v2 mtedx

# Évaluer sur votre dataset de test
python scripts/evaluate_qat.py \
  --model_path outputs/models/whisper-qat-int8-quantized \
  --test_data data/test_sets/your_test.json
```

## 🔬 Détails Techniques

### Fake Quantization

Pendant l'entraînement QAT :
- Les poids et activations sont "fake quantifiés" (simulés)
- Le modèle apprend à fonctionner avec cette contrainte
- Pas de vraie quantization (on garde float32 pour gradients)

### Conversion Finale

Après entraînement :
- Conversion en ONNX quantifié réel
- Int8 : 8 bits par poids/activation
- Int4 : 4 bits (nécessite implémentation custom)

## 📊 Tableau Comparatif pour Publication

Après évaluation, vous aurez :

| Modèle | Format | community-v2 | mtedx | zaion5 | zaion6 | Taille | Vitesse |
|--------|--------|--------------|-------|--------|--------|--------|---------|
| v0.2 | float16 | 9.44 | 8.94 | 29.4 | 26.17 | 100% | 1x |
| v0.2 | int8 (PTQ) | 9.8 | 9.2 | 31.0 | 27.5 | 50% | 2x |
| **v0.3-QAT** | **int8** | **9.5** | **9.0** | **29.8** | **26.5** | **50%** | **2x** |
| v0.3-QAT | int4 | 9.7 | 9.3 | 30.5 | 27.0 | 25% | 3-4x |

*(Valeurs exemple - vos résultats peuvent varier)*

## ✅ Avantages pour Publication

1. **Contribution claire** : Première QAT pour distille Whisper français
2. **Résultats mesurables** : Comparaison directe avec v0.2 et PTQ
3. **Impact pratique** : Déploiement edge/cloud optimisé
4. **Reproducibilité** : Code + datasets publics

## ⚙️ Paramètres Recommandés

### QAT Int8 (Paramètres par Défaut - Optimisé) ⭐

**Version par défaut** (qualité excellente, rapide) :
```yaml
num_epochs: 5  # Suffisant car on part de v0.2 pré-entraîné
learning_rate: 5e-6
batch_size: 8  # Optimal pour GPU
gradient_accumulation: 4
max_samples: 60000  # ~500h de segments 30s
temps_estimé: 2-4h sur GPU moderne
```

**Version Extended** (qualité maximale, plus long) :
```yaml
num_epochs: 10
learning_rate: 5e-6
batch_size: 8
gradient_accumulation: 4
max_samples: 120000  # ~1000h complet
temps_estimé: 6-10h sur GPU moderne
```

**⚠️ Important** : 
- Comme on part de v0.2 déjà entraîné, QAT nécessite **beaucoup moins d'époques** que l'entraînement initial (5 vs 160)
- Pas besoin de tout le dataset : **500h suffisent** (vs 10,000h pour l'entraînement initial)
- **Temps réel : 2-4h sur GPU** avec paramètres optimisés (vs plusieurs jours pour entraînement complet)

### QAT Int4

```yaml
num_epochs: 15  # Plus long car plus difficile
learning_rate: 3e-6  # Encore plus bas
batch_size: 2
gradient_accumulation: 16
```

## 🐛 Troubleshooting

### Erreur "torch.quantization not available"

**Solution** : Installer PyTorch avec support quantization
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Erreur "ONNX conversion failed"

**Solution** : Utiliser Optimum
```bash
pip install optimum[onnxruntime]
```

### Dégradation >1% après QAT

**Causes possibles :**
- Pas assez d'époques (augmenter à 15-20)
- Learning rate trop élevé (réduire)
- Fake quantization mal configurée

**Solution** : Ajuster hyperparamètres et ré-entraîner

## 📝 Checklist Publication

- [ ] Entraînement QAT int8 complété
- [ ] Conversion en modèle quantifié
- [ ] Évaluation sur tous les corpus (community-v2, mtedx, zaion5, zaion6)
- [ ] Comparaison avec v0.2 et PTQ
- [ ] Mesure gains mémoire/vitesse
- [ ] Documentation code + hyperparamètres
- [ ] Publication modèles sur HuggingFace

## 🚀 Quick Start

```bash
# 1. Entraîner QAT (2-4h sur GPU avec paramètres optimisés)
make train-qat-int8

# 2. Convertir en modèle quantifié réel
python scripts/convert_qat_to_quantized.py \
  --model_path outputs/models/whisper-qat-int8/final \
  --output_path outputs/models/whisper-qat-int8-quantized \
  --quantization_type int8

# 3. Évaluer (comparaison avec baseline v0.2)
make evaluate-qat
```

**Temps total workflow : ~3-5 heures** (2-4h training + 30min conversion + 30min évaluation)

---

**En résumé** : Le QAT est **modéré en complexité** et vous pourrez évaluer sur **exactement les mêmes corpus** que v0.2 pour une comparaison directe ! 🎯

