# Solution Simple : Installer torchcodec

## 🎯 Problème

L'erreur indique qu'il manque `torchcodec` pour décoder les données audio.

## ✅ Solution Simple

Sur Vast.ai, exécutez :

```bash
pip install torchcodec
```

Puis relancez l'entraînement :

```bash
bash scripts/train_qat_vast_ai.sh
```

---

## 🔄 Alternative : Utiliser PTQ au lieu de QAT (Plus Simple)

Si vous voulez éviter QAT pour l'instant, vous pouvez utiliser **Post-Training Quantization (PTQ)** qui est plus simple :

```bash
python scripts/quantize_ptq.py \
  --model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
  --quantization_type int8 \
  --output_dir outputs/models/gilbert-whisper-ptq-int8
```

**Avantages PTQ** :
- ✅ Pas besoin d'entraînement (plus rapide)
- ✅ Pas besoin de dataset
- ✅ Fonctionne immédiatement

**Inconvénients PTQ** :
- ⚠️ Légèrement moins bon que QAT (1-2% dégradation vs <0.5%)

---

## 🎯 Recommandation

**Installez simplement torchcodec** :

```bash
pip install torchcodec
bash scripts/train_qat_vast_ai.sh
```

C'est la solution la plus simple et vous obtiendrez les meilleurs résultats avec QAT.

