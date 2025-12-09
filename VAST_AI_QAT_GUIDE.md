# Guide Complet : QAT sur Vast.ai pour Modèle Propriétaire

## 🎯 Objectif

Créer un modèle Whisper optimisé avec **Quantization-Aware Training (QAT)** sur Vast.ai, maximisant :
- ✅ **Performance** : Qualité préservée (<0.5% dégradation WER)
- ✅ **Frugalité** : 2-4x réduction mémoire (int8) ou 4-8x (int4)
- ✅ **Vitesse** : 2-3x plus rapide en inférence

## 📋 Prérequis

1. **Compte Vast.ai** : https://vast.ai
2. **SSH Key** : Clé SSH configurée sur Vast.ai
3. **Repo Git** : Votre projet sur GitHub/GitLab (ou upload manuel)

## 🚀 Setup Rapide (5 minutes)

### Étape 1 : Créer Instance sur Vast.ai

1. Aller sur https://vast.ai
2. **Create** → **GPU Instance**
3. **Sélectionner GPU** :
   - **Recommandé** : RTX 3090, RTX 4090, ou A100 (16GB+ VRAM)
   - **Budget** : RTX 3060 12GB (plus lent mais fonctionne)
   - **Coût** : ~$0.20-0.50/h selon GPU
4. **Template** : PyTorch (ou Ubuntu + CUDA)
5. **Disk Space** : Minimum 100GB (recommandé 200GB+)
6. **Créer l'instance**

### Étape 2 : Se Connecter en SSH

```bash
# Récupérer la commande SSH depuis Vast.ai (dans "Connect")
# Format typique :
ssh root@ssh4.vast.ai -p <PORT> -i ~/.ssh/id_ed25519
```

### Étape 3 : Setup Automatique

Une fois connecté sur Vast.ai, exécuter :

```bash
# Cloner le repo
cd /workspace
git clone <votre-repo-url> finetuning_gilbert
cd finetuning_gilbert

# Lancer setup automatique
bash scripts/setup_vast_ai_qat.sh
```

**C'est tout !** Le script fait :
- ✅ Installation dépendances
- ✅ Configuration environnement
- ✅ Téléchargement datasets (si nécessaire)
- ✅ Lancement entraînement QAT optimisé

## 📊 Configuration Optimisée

### Paramètres pour Performance/Frugalité/Vitesse

Le script utilise ces paramètres optimisés :

```yaml
# Performance (qualité)
- num_epochs: 5  # Suffisant car modèle déjà pré-entraîné
- learning_rate: 5e-6  # Conservateur pour préserver qualité
- max_samples: 60000  # ~500h (vs 1000h+ complet)

# Frugalité (mémoire)
- per_device_batch_size: 8  # Optimisé pour GPU 16-24GB
- gradient_accumulation_steps: 4  # Équivalent batch_size 32
- fp16: true  # Réduit mémoire de 50%

# Vitesse (inférence)
- quantization_type: int8  # 2-3x plus rapide que float16
- format: onnx  # Optimisé pour inférence
```

## 🔧 Scripts Disponibles

### 1. Setup Automatique

```bash
bash scripts/setup_vast_ai_qat.sh
```

**Fait** :
- Vérifie GPU et espace disque
- Installe dépendances (transformers, optimum, etc.)
- Configure cache HuggingFace sur `/workspace` (plus d'espace)
- Prépare environnement

### 2. Entraînement QAT Optimisé

```bash
# Option A : Script automatique (recommandé)
bash scripts/train_qat_vast_ai.sh

# Option B : Commande manuelle
python scripts/train_qat_optimized.py \
  --base_model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --quantization_type int8 \
  --output_dir outputs/models/gilbert-whisper-qat-int8 \
  --max_samples 60000 \
  --num_epochs 5 \
  --per_device_batch_size 8
```

### 3. Conversion en Modèle Quantifié

Après entraînement QAT :

```bash
python scripts/convert_qat_to_quantized.py \
  --model_path outputs/models/gilbert-whisper-qat-int8/final \
  --output_path outputs/models/gilbert-whisper-qat-int8-quantized \
  --quantization_type int8 \
  --format onnx
```

### 4. Benchmark Performance

```bash
python scripts/benchmark_quantized.py \
  --model_path outputs/models/gilbert-whisper-qat-int8-quantized \
  --device cuda \
  --num_runs 20
```

## ⏱️ Temps Estimé

| Étape | Temps | Description |
|-------|-------|-------------|
| **Setup** | 5-10 min | Installation dépendances |
| **Téléchargement datasets** | 10-30 min | Si pas déjà téléchargés |
| **Entraînement QAT** | **2-4h** | Sur GPU moderne (RTX 3090+) |
| **Conversion quantifiée** | 5-10 min | ONNX + quantization |
| **Benchmark** | 5 min | Tests performance |
| **Total** | **3-5h** | De bout en bout |

**Coût estimé** : $0.60-2.00 (selon GPU et durée)

## 📈 Résultats Attendus

### Métriques Cibles

| Métrique | Baseline (v0.2) | QAT int8 | Amélioration |
|----------|----------------|----------|--------------|
| **WER** | Référence | +0.3-0.5% | ✅ Minimal |
| **Taille** | 1.51 GB | **0.75 GB** | ✅ **-50%** |
| **VRAM** | 1.57 GB | **0.8 GB** | ✅ **-49%** |
| **Vitesse** | Baseline | **2-3x** | ✅ **+200%** |
| **Latence** | 0.053s | **0.02-0.03s** | ✅ **-40%** |

### Comparaison Formats

| Format | Taille | VRAM | Vitesse | Qualité |
|--------|--------|------|---------|---------|
| **FP16** (baseline) | 1.51 GB | 1.57 GB | 1x | 100% |
| **int8 (QAT)** | 0.75 GB | 0.8 GB | 2-3x | 99.5% |
| **int4 (QAT)** | 0.38 GB | 0.4 GB | 4-5x | 98% |

## 🎯 Workflow Complet

### Phase 1 : Setup (10 min)

```bash
# Sur Vast.ai
cd /workspace
git clone <votre-repo> finetuning_gilbert
cd finetuning_gilbert
bash scripts/setup_vast_ai_qat.sh
```

### Phase 2 : Entraînement QAT (2-4h)

```bash
# Lancer entraînement (peut tourner en arrière-plan)
nohup bash scripts/train_qat_vast_ai.sh > training.log 2>&1 &

# Suivre les logs
tail -f training.log
```

### Phase 3 : Conversion (10 min)

```bash
# Après entraînement terminé
python scripts/convert_qat_to_quantized.py \
  --model_path outputs/models/gilbert-whisper-qat-int8/final \
  --output_path outputs/models/gilbert-whisper-qat-int8-quantized \
  --quantization_type int8
```

### Phase 4 : Évaluation (15 min)

```bash
# Benchmark performance
python scripts/benchmark_quantized.py \
  --model_path outputs/models/gilbert-whisper-qat-int8-quantized

# Évaluation qualité (WER/CER)
python scripts/evaluate_wer.py \
  --model outputs/models/gilbert-whisper-qat-int8-quantized \
  --dataset facebook/multilingual_librispeech \
  --dataset_config french \
  --split test \
  --max_samples 100
```

### Phase 5 : Sauvegarde (5 min)

```bash
# Option A : Upload vers HuggingFace (recommandé)
huggingface-cli login
huggingface-cli upload <votre-username>/gilbert-whisper-qat-int8 \
  outputs/models/gilbert-whisper-qat-int8-quantized

# Option B : Télécharger localement
# Depuis votre machine locale
scp -r root@<vast-ip>:/workspace/finetuning_gilbert/outputs/models/gilbert-whisper-qat-int8-quantized ./
```

## 🔍 Monitoring

### Pendant l'Entraînement

```bash
# Voir logs en temps réel
tail -f training.log

# Vérifier utilisation GPU
watch -n 1 nvidia-smi

# Vérifier espace disque
df -h /workspace
```

### Métriques à Surveiller

- **Loss** : Doit diminuer progressivement
- **WER (eval)** : Doit rester proche du baseline (<1% dégradation)
- **GPU Utilisation** : Doit être >80% pendant training
- **VRAM** : Ne pas dépasser capacité GPU

## 🐛 Troubleshooting

### Problème : "Out of Memory"

**Solution** :
```bash
# Réduire batch size
python scripts/train_qat_optimized.py \
  --per_device_batch_size 4 \  # Au lieu de 8
  --gradient_accumulation_steps 8  # Compenser
```

### Problème : "No space left on device"

**Solution** :
```bash
# Nettoyer cache
bash scripts/cleanup_disk.sh

# Utiliser /workspace pour cache HuggingFace
export HF_HOME=/workspace/.hf_home
export TRANSFORMERS_CACHE=/workspace/.hf_home
```

### Problème : "CUDA out of memory"

**Solution** :
```bash
# Réduire batch size et activer gradient checkpointing
python scripts/train_qat_optimized.py \
  --per_device_batch_size 2 \
  --gradient_checkpointing
```

## 📝 Notes Importantes

### Pour Modèle Propriétaire

1. **Nom du modèle** : Utiliser `gilbert-whisper-qat-int8` (ou votre nom)
2. **Licence** : Spécifier dans model card (MIT si basé sur v0.2)
3. **Crédits** : Mentionner base `bofenghuang/whisper-large-v3-distil-fr-v0.2`

### Optimisations Incluses

- ✅ **FP16 training** : Réduit mémoire
- ✅ **Gradient accumulation** : Simule batch size plus grand
- ✅ **Optimized datasets** : Sous-ensemble pour accélérer
- ✅ **ONNX export** : Format optimisé inférence
- ✅ **Cache management** : Utilise `/workspace` (plus d'espace)

## 🎓 Prochaines Étapes

Après QAT réussi :

1. **Publier sur HuggingFace** : Modèle quantifié prêt à l'emploi
2. **Benchmark complet** : Comparer avec baseline sur tous datasets
3. **Documentation** : Créer model card avec métriques
4. **Déploiement** : Intégrer dans votre application

## 💡 Astuces

- **Sauvegarder checkpoints** : Le script sauvegarde automatiquement
- **Resume training** : Si interrompu, peut reprendre depuis checkpoint
- **Multi-GPU** : Si disponible, activer avec `--num_gpus`
- **TensorBoard** : Logs disponibles dans `outputs/logs/`

---

**Questions ?** Voir `GUIDE_QAT.md` pour détails techniques.

