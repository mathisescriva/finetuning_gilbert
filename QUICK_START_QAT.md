# Quick Start : Lancer QAT depuis CLI

## 🎯 La Solution la Plus Simple

### Option 1 : RunPod (Recommandé) ⭐

**1. Créer compte sur https://runpod.io**

**2. Lancer une instance GPU** :
- Template : PyTorch
- GPU : RTX 3090 ou A100 (selon budget)
- Coût : ~$0.20-0.40/h

**3. Se connecter en SSH** et exécuter :

```bash
# Clone votre repo
git clone <votre-repo-url>
cd finetuning_gilbert

# Setup automatique + entraînement
bash setup_and_train.sh
```

**C'est tout !** Le script fait tout automatiquement :
- ✅ Installe dépendances
- ✅ Télécharge dataset
- ✅ Lance entraînement QAT (2-4h)

**Coût total : ~$0.80-1.60**

---

### Option 2 : Google Colab (Gratuit/Pro)

**1. Ouvrir https://colab.research.google.com**

**2. Nouveau notebook → Runtime → Change runtime type → GPU**

**3. Exécuter ces cellules** :

```python
# Cellule 1 : Installer dépendances
!pip install -q transformers datasets accelerate librosa soundfile jiwer optimum[onnxruntime] torch torchaudio

# Cellule 2 : Cloner repo (ou uploader manuellement)
!git clone <votre-repo-url>
%cd finetuning_gilbert

# Cellule 3 : Setup
!bash setup_and_train.sh
```

**Limite** : 12h max (free) ou illimité (Pro $10/mois)

---

### Option 3 : Local (Si GPU disponible)

```bash
# 1. Vérifier GPU
nvidia-smi

# 2. Lancer setup automatique
bash setup_and_train.sh
```

---

## 📋 Commandes CLI Essentielles

### Vérifier l'environnement

```bash
# Vérifier Python
python3 --version  # Doit être 3.8+

# Vérifier GPU
nvidia-smi  # Doit afficher votre GPU

# Vérifier PyTorch + CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Setup manuel (si script ne fonctionne pas)

```bash
# 1. Installer dépendances
pip install -r requirements.txt
pip install optimum[onnxruntime]

# 2. Télécharger dataset
python scripts/download_datasets.py --datasets common_voice --max_samples 60000

# 3. Lancer entraînement
make train-qat-int8
```

### Monitoring pendant l'entraînement

```bash
# Voir les logs
tail -f outputs/logs/trainer_logs.txt

# Ou avec TensorBoard (si installé)
tensorboard --logdir outputs/logs
```

---

## 🐛 Problèmes Courants

### "CUDA out of memory"

```bash
# Réduire batch size
python scripts/train_qat.py \
  --per_device_batch_size 4  # Au lieu de 8
```

### "Module not found"

```bash
# Réinstaller dépendances
pip install -r requirements.txt --upgrade
```

### Dataset non trouvé

```bash
# Télécharger manuellement
python scripts/download_datasets.py --datasets common_voice
```

---

## ✅ Après Entraînement

```bash
# 1. Convertir en quantifié
python scripts/convert_qat_to_quantized.py \
  --model_path outputs/models/whisper-qat-int8/final \
  --output_path outputs/models/whisper-qat-int8-quantized \
  --quantization_type int8

# 2. Évaluer
make evaluate-qat
```

---

## 💡 Recommandation

**Pour votre cas** : Utilisez **RunPod** ou **Vast.ai**

- Setup en 5 minutes
- GPU pas cher
- Accès SSH direct
- Script automatique fait tout

**Commande complète** :
```bash
git clone <repo> && cd finetuning_gilbert && bash setup_and_train.sh
```

C'est tout ce qu'il faut ! 🚀

