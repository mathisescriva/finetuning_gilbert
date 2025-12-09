# Solution Finale : QAT sans Problèmes de Disque

## 🎯 Problème

Le disque est plein, même le streaming télécharge des métadonnées qui remplissent le cache.

## ✅ Solution : Nettoyer le cache AVANT + Utiliser dataset minimal

### Sur Vast.ai, exécutez dans cet ordre :

```bash
# 1. NETTOYER TOUT LE CACHE HUGGINGFACE
rm -rf /workspace/.hf_home/hub/*
rm -rf ~/.cache/huggingface/*

# 2. Nettoyer pip et autres caches
pip cache purge
rm -rf /tmp/*

# 3. Vérifier espace
df -h /workspace

# 4. Mettre à jour scripts
cd /workspace/finetuning_gilbert
git pull

# 5. Lancer avec dataset TRÈS petit (1000 échantillons seulement)
python scripts/train_qat_simple.py \
  --base_model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --output_dir outputs/models/gilbert-whisper-qat-int8 \
  --max_samples 1000 \
  --num_epochs 3 \
  --batch_size 8 \
  --learning_rate 1e-5
```

## 🔄 Alternative : Utiliser PTQ Directement

Si le problème persiste, **utilisez PTQ directement** (pas besoin de dataset) :

```bash
cd /workspace/finetuning_gilbert

# Quantifier directement (pas besoin d'entraînement ni de dataset)
python scripts/quantize_ptq.py \
  --model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --quantization_type int8 \
  --output_dir outputs/models/gilbert-whisper-ptq-int8
```

**Résultat** :
- ✅ Fonctionne immédiatement (5-10 min)
- ✅ Pas besoin de dataset
- ✅ Pas de problème de disque
- ⚠️ Qualité : ~1-2% dégradation (vs <0.5% avec QAT, mais acceptable)

---

## 💡 Pour Vraie QAT : Utiliser Vos Propres Données

Si vous avez des données audio + transcripts :

```bash
# Créer un fichier JSON simple
# data/my_data.json
[
  {"audio": "path/to/audio1.wav", "text": "transcription 1"},
  {"audio": "path/to/audio2.wav", "text": "transcription 2"},
  ...
]

# Utiliser ce dataset local
python scripts/train_qat_simple.py \
  --base_model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --output_dir outputs/models/gilbert-whisper-qat-int8 \
  --train_data data/my_data.json \
  --eval_data data/my_data.json \
  --max_samples 1000
```

---

## 🎯 Recommandation Immédiate

**Pour avancer rapidement** : Utilisez **PTQ** directement. Vous obtiendrez votre modèle quantifié en 5-10 minutes sans problème.

