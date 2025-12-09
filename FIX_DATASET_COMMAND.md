# Solution : Dataset Common Voice non disponible

## 🔧 Problème

Common Voice n'est pas accessible facilement avec les versions récentes de HuggingFace datasets.

## ✅ Solution : Utiliser MLS (Multilingual LibriSpeech)

MLS est plus stable et toujours disponible. J'ai modifié les scripts pour utiliser MLS automatiquement.

## 🚀 Commandes à exécuter sur Vast.ai

### Option 1 : Relancer avec MLS (automatique)

Les scripts ont été mis à jour pour utiliser MLS. Il suffit de relancer :

```bash
cd /workspace/finetuning_gilbert
git pull  # Mettre à jour les scripts
bash scripts/train_qat_vast_ai.sh
```

### Option 2 : Utiliser MLS directement (manuel)

```bash
cd /workspace/finetuning_gilbert

python scripts/train_qat_optimized.py \
  --base_model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --train_data facebook/multilingual_librispeech \
  --eval_data facebook/multilingual_librispeech \
  --quantization_type int8 \
  --output_dir outputs/models/gilbert-whisper-qat-int8 \
  --num_epochs 5 \
  --max_samples 60000 \
  --per_device_batch_size 16 \
  --learning_rate 5e-6
```

## 📊 À propos de MLS

- ✅ **Disponible** : Toujours accessible sur HuggingFace
- ✅ **Français** : Version française de qualité
- ✅ **Stable** : Pas de problèmes de versions
- ✅ **Compatible** : Format standard HuggingFace

## 🔄 Mettre à jour les scripts

Si vous voulez mettre à jour les scripts depuis GitHub :

```bash
cd /workspace/finetuning_gilbert
git pull origin main
```

Les modifications incluent :
- Utilisation automatique de MLS au lieu de Common Voice
- Meilleure gestion des erreurs de chargement
- Fallback automatique vers MLS

---

**Relancez simplement** : `bash scripts/train_qat_vast_ai.sh` après `git pull` !

