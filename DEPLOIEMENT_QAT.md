# Déploiement Simple : Entraînement QAT depuis CLI

## 🎯 Options Simples

### Option 1 : RunPod / Vast.ai (Recommandé) ⭐

**Pourquoi** : GPU pas cher (~$0.20-0.40/h), setup simple

```bash
# 1. Créer compte sur runpod.io ou vast.ai
# 2. Lancer instance GPU (RTX 3090, A100, etc.)

# 3. Sur l'instance, cloner votre repo
git clone <votre-repo>
cd finetuning_gilbert

# 4. Installer dépendances
pip install -r requirements.txt
pip install optimum[onnxruntime]

# 5. Lancer l'entraînement
make train-qat-int8
```

**Coût** : ~$0.80-1.60 pour 2-4h d'entraînement

---

### Option 2 : Google Colab Pro

**Pourquoi** : Gratuit (version free) ou $10/mois (Pro), GPU inclus

1. Ouvrir Google Colab
2. Créer nouveau notebook
3. Upload votre projet ou cloner depuis GitHub
4. Lancer les cellules :

```python
# Installer dépendances
!pip install -q transformers datasets accelerate librosa soundfile jiwer optimum[onnxruntime]
!pip install -q torch torchaudio

# Cloner repo (ou uploader fichiers)
!git clone <votre-repo-url>
%cd finetuning_gilbert

# Lancer entraînement
!python scripts/train_qat.py \
  --base_model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --train_data <votre-dataset> \
  --eval_data <votre-dataset> \
  --quantization_type int8
```

**Limite** : Colab free = 12h max, Pro = plus long

---

### Option 3 : AWS / GCP / Azure (Si compte existant)

**AWS EC2 avec GPU** :

```bash
# Lancer instance (g4dn.xlarge ou p3.2xlarge)
aws ec2 run-instances \
  --image-id ami-xxxxx \
  --instance-type g4dn.xlarge \
  --key-name your-key

# SSH dans l'instance
ssh -i your-key.pem ubuntu@<ip>

# Sur l'instance
git clone <repo>
cd finetuning_gilbert
pip install -r requirements.txt
make train-qat-int8
```

---

### Option 4 : Local (Si GPU disponible)

```bash
# Vérifier GPU
nvidia-smi

# Installer dépendances
pip install -r requirements.txt
pip install optimum[onnxruntime]

# Lancer
make train-qat-int8
```

---

## 🚀 Script d'Setup Automatique

J'ai créé un script pour setup automatique :

```bash
# Sur n'importe quelle machine (RunPod, Colab, local, etc.)
bash setup_and_train.sh
```

Le script va :
1. Installer toutes les dépendances
2. Télécharger les datasets si nécessaire
3. Lancer l'entraînement QAT
4. Sauvegarder les résultats

---

## 📋 Checklist Rapide

### Sur RunPod/Vast.ai (Recommandé)

```bash
# 1. Connecter à l'instance (SSH)
ssh root@<ip>

# 2. Setup
git clone <votre-repo>
cd finetuning_gilbert
pip install -r requirements.txt
pip install optimum[onnxruntime]

# 3. Télécharger dataset (si nécessaire)
python scripts/download_datasets.py --datasets common_voice

# 4. Lancer entraînement
make train-qat-int8

# 5. Sauvegarder résultats (optionnel: upload vers S3/GCS)
```

### Sur Google Colab

1. Nouveau notebook
2. Runtime → Change runtime type → GPU
3. Exécuter les cellules (voir Option 2 ci-dessus)

---

## 💡 Recommandation Finale

**Pour simplicité maximale** : **RunPod** ou **Vast.ai**

- ✅ Setup en 5 minutes
- ✅ GPU puissant pas cher
- ✅ Accès SSH direct
- ✅ Pas de limite de temps (vs Colab)
- ✅ Coût : ~$1-2 total

Voulez-vous que je crée un script d'auto-setup complet ?

