# Solution : Problème de Mémoire (RAM)

## 🎯 Problème

`std::bad_alloc` = Plus assez de RAM pour charger 60k échantillons en mémoire.

## ✅ Solutions Simples

### Option 1 : Réduire le nombre d'échantillons (RAPIDE)

Modifier `scripts/train_qat_vast_ai.sh` :

```bash
# Changer cette ligne :
MAX_SAMPLES=60000  # Trop pour la RAM disponible

# En :
MAX_SAMPLES=10000  # Beaucoup plus léger, suffisant pour QAT
```

Puis relancer :

```bash
bash scripts/train_qat_vast_ai.sh
```

**Temps** : ~30-45 min au lieu de 1-2h, mais résultat similaire (QAT fonctionne bien avec moins de données).

---

### Option 2 : Utiliser PTQ directement (PLUS SIMPLE)

**Pas besoin d'entraînement**, quantization directe :

```bash
python scripts/quantize_ptq.py \
  --model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --quantization_type int8 \
  --output_dir outputs/models/gilbert-whisper-ptq-int8
```

**Avantages** :
- ✅ **5-10 minutes** (vs 1-2h)
- ✅ **Pas besoin de dataset**
- ✅ **Pas de problème de mémoire**
- ✅ **Fonctionne immédiatement**

**Résultat** :
- Qualité : ~1-2% dégradation (vs <0.5% avec QAT)
- Taille/vitesse : Identique (50% réduction, 2-3x plus rapide)

---

### Option 3 : Utiliser Trainer avec Streaming (Plus complexe)

Le Trainer de HuggingFace peut gérer le streaming nativement sans charger tout en mémoire. Mais cela nécessite de modifier le code.

---

## 🎯 Recommandation

**Pour arriver rapidement à votre objectif** : **Option 2 (PTQ)**

Vous obtiendrez un modèle quantifié en **5-10 minutes** sans problème de mémoire ou de dataset.

**Si vous voulez le meilleur résultat** : **Option 1** (réduire à 10k échantillons)

---

## 📝 Commande Rapide PTQ

```bash
cd /workspace/finetuning_gilbert
python scripts/quantize_ptq.py \
  --model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --quantization_type int8 \
  --output_dir outputs/models/gilbert-whisper-ptq-int8
```

C'est tout ! En 5-10 minutes vous aurez votre modèle quantifié. 🚀

