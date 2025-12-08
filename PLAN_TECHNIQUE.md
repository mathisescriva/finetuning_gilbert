# Plan Technique : Modèle Whisper Spécialisé Réunions

## 1. Analyse du Modèle de Base

### 1.1 Caractéristiques de `bofenghuang/whisper-large-v3-distil-fr-v0.2`

**Architecture :**
- Encodeur : Identique à `whisper-large-v3` (conservation des capacités d'encodage)
- Décodeur : Réduit à 2 couches (vs 6 dans large-v3) via distillation
- Paramètres : ~50% de réduction par rapport à large-v3
- Vitesse : 5-6x plus rapide que large-v3

**Méthode de distillation :**
- "Patient teacher" : le teacher produit des prédictions sur plusieurs générations
- Augmentations agressives : amélioration de la robustesse
- Segments de 30s : maintien des capacités long-form
- Moins d'hallucinations en long-form grâce à la distillation patiente

**Spécificité français :**
- Entraîné spécifiquement sur données françaises
- Optimisé pour accents et variations régionales

### 1.2 Forces pour le Cas d'Usage Réunions

✅ **Robustesse :**
- Bonne gestion des accents français variés
- Résistance au bruit ambiant (grâce aux augmentations)
- Capacité long-form (segments de 30s) adaptée aux réunions longues
- Moins d'hallucinations que large-v3 (crucial pour CR sérieux)

✅ **Performance/Frugalité :**
- Meilleur compromis que large-v3 complet
- Compatible avec transformers, faster-whisper, whisper.cpp
- Déploiement possible sur GPU moyen (16-24 Go)

✅ **Qualité française :**
- Spécialisé français (pas généraliste multilingue)
- Meilleure reconnaissance des spécificités linguistiques françaises

### 1.3 Faiblesses / Manques pour Réunions

❌ **Spécialisation domaine :**
- Pas spécifiquement entraîné sur données de réunions
- Vocabulaire généraliste, pas optimisé pour jargon métier
- Noms propres moins bien reconnus (participants, entreprises)

❌ **Frugalité encore limitée :**
- Peut être trop lourd pour edge devices
- Pas quantifié par défaut (int8/int4 possible)
- Encore coûteux en VRAM pour petites machines

❌ **Patterns spécifiques réunions :**
- Chevauchements de parole (même avec diarisation)
- Hésitations et reformulations ("euh", "voilà", etc.)
- Longs contextes (30-120 minutes) avec cohérence

## 2. Stratégie Technique

### 2.1 Évaluation de Base

**Objectif :** Établir une baseline claire sur des données de réunions.

**Métriques :**
- WER (Word Error Rate) global
- CER (Character Error Rate) global
- WER spécialisé :
  - Noms propres (participants, entreprises)
  - Acronymes techniques
  - Termes métier fréquents
  - Nombres et dates
- Latence (secondes/minute audio)
- Mémoire (VRAM/RAM)

**Jeux de données :**
- Dev set : 20-30 réunions variées
- Test set : 10-15 réunions (non vues pendant l'entraînement)

### 2.2 Fine-tuning / Adaptation Domaine "Réunions"

**Choix : Approche Hybride (Fine-tuning + LoRA)**

**Phase 1 : Fine-tuning Full sur données réunions**
- **Données :** Mix dataset public français (Common Voice, MLS) + nos données réunions
- **Augmentations :**
  - Bruit de fond bureau (5-15 dB SNR)
  - Écho de salle (simulation RIR)
  - Variations de qualité micro (visio, téléphone, studio)
  - Compression audio (codecs variés)
- **Hyperparamètres :**
  - Learning rate : 1e-5 (encoder frozen initialement), puis 5e-6 (full)
  - Batch size : 8-16 (selon GPU)
  - Epochs : 3-5 avec early stopping
  - Scheduler : Warmup cosine (10% warmup)
  - Weight decay : 0.01

**Phase 2 : LoRA pour spécialisation fine**
- **Placement :** Décodeur uniquement (où se fait la génération de vocabulaire)
- **Rank :** 8-16 (trade-off perf/frugalité)
- **Alpha :** 16-32
- **Avantages :**
  - Adaptation légère par client/domaine
  - Plug & play de LoRA spécialisés
  - Réduction mémoire (30-50% paramètres entraînables)

**Justification :**
- Fine-tuning full garantit une base solide sur réunions
- LoRA permet spécialisation fine sans sur-apprentissage
- Modularité pour déploiements multiples domaines

### 2.3 Frugalité : Distillation + Quantization

**A. Distillation Complémentaire "Réunions"**

**Teacher :** Notre modèle fine-tuné (ou whisper-large-v3-french si besoin)
**Student :** Version encore plus légère

**Stratégie :**
- Pseudo-labelling : générer labels sur dataset non annoté réunions
- Patient teacher : utiliser teacher avec plusieurs beams
- Data augmentation agressive
- Objectif : réduire encore 20-30% paramètres si possible

**B. Quantization**

**Option 1 : Post-Training Quantization (PTQ)**
- **Int8 :** Qualité préservée (~1% WER dégradation), 2x réduction taille
- **Int4 :** Plus agressif (~3-5% WER), 4x réduction

**Option 2 : Quantization-Aware Training (QAT)**
- Entraîner avec faux quantifiés
- Meilleure qualité que PTQ (dégradation <1%)
- Compatible avec distillation

**Recommandation :** PTQ int8 pour production, QAT int8 pour qualité maximale

**C. Optimisation Inférence**

**Stacks recommandées :**
1. **Production GPU :** `faster-whisper` (CTranslate2 backend, très rapide)
2. **Production CPU :** `whisper.cpp` (optimisé CPU, peut être quantifié)
3. **Edge/Embedded :** `transformers` + `torch.compile` ou ONNX

**Paramètres optimisés :**
- `chunk_length_s` : 30 (optimal pour long-form)
- `beam_size` : 5 (bon compromis)
- `best_of` : 5
- `temperature` : 0.0 (déterministe)
- `log_prob_threshold` : -1.0 (filtre faible confiance)
- `compression_ratio_threshold` : 2.4 (détecte répétitions/hallucinations)

### 2.4 Spécialisation Vocabulaire & Lexiques

**A. Fine-tuning Ciblé**
- Sur-échantillonnage de segments contenant noms propres/acronymes
- Création d'un dataset enrichi avec termes métier

**B. Language Model Biasing**
- Shallow fusion avec un petit LM spécialisé réunions
- Boost de probabilité pour noms participants (si liste disponible)

**C. Post-processing Lexique**
- Correction automatique de patterns d'erreurs fréquents
- Mapping noms propres mal transcrits (si lexique connu)

**Recommandation :** Combinaison fine-tuning ciblé + post-processing (plus pragmatique)

## 3. Architecture Technique Proposée

### 3.1 Modèles Finaux (2 Variantes)

**Modèle 1 : "Production Réunions"**
- Base : `bofenghuang/whisper-large-v3-distil-fr-v0.2` fine-tuné réunions
- Quantization : Int8 PTQ
- Target : GPU 16-24 Go, latence < 0.1x real-time
- Usage : Serveur production, qualité optimale

**Modèle 2 : "Edge Réunions" (optionnel)**
- Base : Modèle 1 distillé supplémentaire
- Quantization : Int4 ou Int8 très agressive
- Target : CPU/mobile, latence < 0.3x real-time
- Usage : On-prem, edge devices, latence acceptable

### 3.2 Pipeline de Traitement

```
Audio (réunion) 
  → Pré-processing (normalisation, VAD optionnel)
  → Segmentation (30s chunks, overlap 1s)
  → Transcription (modèle spécialisé)
  → Post-processing (lexique, correction patterns)
  → Concatenation (merge chunks)
  → Sortie (texte brut ou structuré)
```

## 4. Protocole d'Évaluation

### 4.1 Métriques Comparatives

| Modèle | WER Global | WER Noms Propres | WER Acronymes | Latence (s/min) | VRAM (Go) |
|--------|------------|------------------|---------------|-----------------|-----------|
| whisper-large-v3 (ref) | Baseline | Baseline | Baseline | ~20 | ~10 |
| distil-fr-v0.2 (baseline) | Baseline | Baseline | Baseline | ~4 | ~5 |
| **Production Réunions** | **Target: -15%** | **Target: -25%** | **Target: -20%** | **<6** | **<6** |
| Edge Réunions (opt) | Target: -10% | Target: -15% | Target: -15% | <18 | <3 |

### 4.2 Analyse Qualitative

- Types d'erreurs corrigées vs baseline
- Cas limites identifiés (bruit extrême, accents rares)
- Robustesse longue durée (30-120 min)

## 5. Timeline & Livrables

1. **Semaine 1-2 :** Évaluation baseline + setup infrastructure
2. **Semaine 3-4 :** Fine-tuning réunions + LoRA
3. **Semaine 5-6 :** Distillation + Quantization
4. **Semaine 7 :** Évaluation complète + documentation

**Livrables :**
- ✅ Checkpoints modèles (Production + Edge)
- ✅ Scripts reproductibles
- ✅ Guide d'intégration
- ✅ Rapport d'évaluation

## 6. Risques & Mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Overfitting données réunions | Perte robustesse générale | Mix dataset public + réunions, early stopping |
| Quantization trop agressive | Perte qualité inacceptable | Tests incrémentaux, QAT si nécessaire |
| Latence trop élevée | Non viable production | Optimisation inférence, faster-whisper |
| Données réunions insuffisantes | Amélioration limitée | Pseudo-labelling, augmentations agressives |

## 7. Next Steps & Améliorations Futures

- **Diarisation jointe :** Modèle ASR + speaker diarization end-to-end
- **Adaptation par secteur :** LoRA spécialisés (tech, finance, santé, etc.)
- **Multilingue réunions :** Extension à anglais/autres langues si besoin
- **Temps réel :** Streaming transcription avec latence <2s
- **Fine-tuning continu :** Mécanisme d'apprentissage en production (avec supervision)

