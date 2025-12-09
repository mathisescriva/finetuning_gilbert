# Quick Start : RTX 5090 sur Vast.ai

## 🚀 Excellente carte ! RTX 5090 = Ultra Rapide

Avec la RTX 5090, vous pouvez vous attendre à :
- ✅ **Temps d'entraînement** : ~1-1.5h (au lieu de 2-4h)
- ✅ **Batch size optimisé** : 16 (au lieu de 8)
- ✅ **Performance maximale** : Profite de la dernière génération

## 📋 Commandes à Exécuter (dans l'ordre)

### 1. Cloner le repo

```bash
cd /workspace
git clone https://github.com/mathisescriva/finetuning_gilbert.git finetuning_gilbert
cd finetuning_gilbert
```

### 2. Vérifier la GPU

```bash
nvidia-smi
# Devrait afficher RTX 5090 et la VRAM disponible
```

### 3. Lancer le setup

```bash
bash scripts/setup_vast_ai_qat.sh
```

### 4. Lancer l'entraînement QAT (optimisé RTX 5090)

```bash
bash scripts/train_qat_vast_ai.sh
```

**Le script utilise automatiquement batch_size=16 pour RTX 5090 !**

## ⚡ Option : Encore Plus Rapide

Si vous voulez **maximiser la vitesse** et avez assez de VRAM :

### Option A : Batch size 32 (très rapide)

Modifier `scripts/train_qat_vast_ai.sh` :
```bash
BATCH_SIZE=32
GRADIENT_ACCUMULATION=1
```

### Option B : Plus d'échantillons (meilleure qualité)

Modifier `scripts/train_qat_vast_ai.sh` :
```bash
MAX_SAMPLES=100000  # Au lieu de 60000
```

## 📊 Temps Estimé

- **Setup** : 5-10 min
- **Téléchargement datasets** : 10-30 min (si nécessaire)
- **Entraînement QAT** : **1-1.5h** avec RTX 5090
- **Conversion quantifiée** : 5-10 min
- **Total** : ~2h de bout en bout

## 🎯 Monitoring

Pendant l'entraînement, dans un autre terminal SSH :

```bash
# Voir utilisation GPU
watch -n 1 nvidia-smi

# Voir les logs en temps réel
tail -f outputs/models/gilbert-whisper-qat-int8/training.log
```

## ✅ Après l'Entraînement

```bash
# Convertir en modèle quantifié
python scripts/convert_qat_to_quantized.py \
  --model_path outputs/models/gilbert-whisper-qat-int8/final \
  --output_path outputs/models/gilbert-whisper-qat-int8-quantized \
  --quantization_type int8

# Benchmark
python scripts/benchmark_quantized.py \
  --model_path outputs/models/gilbert-whisper-qat-int8-quantized
```

---

**Bonne chance ! Avec la RTX 5090, ça va être rapide ! ⚡**

