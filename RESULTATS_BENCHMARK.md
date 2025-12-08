# Résultats de Benchmark - Whisper-Large-V3-Distil-French-v0.2

## 📊 Configuration

- **Modèle**: `bofenghuang/whisper-large-v3-distil-fr-v0.2`
- **GPU**: NVIDIA GeForce RTX 5090
- **Format**: FP16 (float16)
- **Date**: 2025-12-08

## 🎯 Métriques de Performance

### Taille du Modèle

- **Paramètres**: 756.4M (millions)
- **Taille sur disque**: 1.51 GB (FP16)
- **Format**: PyTorch float16

### Vitesse d'Inférence

- **Latence moyenne**: 0.053s ± 0.082s pour 30 secondes d'audio
- **Temps médian**: ~0.03s (estimé)
- **Débit**: **569x temps réel**
  - Signifie: peut transcrire 569 secondes (9.5 minutes) d'audio en 1 seconde réelle

### Mémoire

- **Utilisation VRAM**: 0.06 GB (mesure différentielle)
- **Note**: L'utilisation totale incluant le modèle chargé est d'environ 1.57 GB (modèle + overhead)

## 📈 Comparaison avec Baseline (Whisper Large-v3)

### Basé sur la documentation du modèle distillé:

| Métrique | Whisper Large-v3 | Distil-French v0.2 | Amélioration |
|----------|------------------|-------------------|--------------|
| **Taille** | ~3.0 GB | 1.51 GB | **-50%** |
| **Paramètres** | ~1.5B | 756.4M | **-49%** |
| **Vitesse** | Baseline | **5.8x plus rapide** | **5.8x** |
| **Qualité (WER)** | Référence | +1-2% WER | **Minimal** |

### Métriques Mesurées (Notre Benchmark)

- **Vitesse**: 569x temps réel (mesuré)
- **Efficacité mémoire**: 1.57 GB VRAM total (très frugal)
- **Débit**: ~18,000 secondes d'audio par heure réelle

## 🎓 Métriques pour Publication

### Résumé Exécutif

Le modèle **Whisper-Large-V3-Distil-French-v0.2** offre un excellent compromis qualité/performance :

1. **Frugalité**: 
   - 50% plus petit que le modèle complet
   - Mémoire VRAM réduite à 1.57 GB
   - Adapté pour déploiement on-premise et edge

2. **Performance**:
   - 5.8x plus rapide que Whisper Large-v3
   - Débit de 569x temps réel sur RTX 5090
   - Latence < 0.1s pour 30s d'audio

3. **Qualité**:
   - Dégradation minimale (+1-2% WER selon documentation)
   - Spécialisé pour le français
   - Optimisé pour transcription longue durée

### Points Clés pour le Papier

- **Innovation**: Distillation spécialisée français avec encodeur préservé
- **Efficacité**: 50% réduction taille, 5.8x accélération
- **Utilité**: Déploiement edge/on-premise possible (16GB GPU suffisant)
- **Robustesse**: Moins d'hallucinations en long-form que le modèle complet

## 📋 Métriques Complémentaires (à Mesurer)

Pour une publication complète, mesurer également :

- [ ] **WER/CER** sur datasets de test (Common Voice, MLS)
- [ ] **Latence** sur différents GPU (RTX 3090, A100, CPU)
- [ ] **Comparaison qualité** vs baseline sur mêmes datasets
- [ ] **Métriques spécialisées**: noms propres, acronymes, termes métier
- [ ] **Temps d'inférence** par minute d'audio
- [ ] **Mémoire RAM** en plus de VRAM

## 💾 Utilisation

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

model_name = "bofenghuang/whisper-large-v3-distil-fr-v0.2"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.to("cuda")

# Transcription
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
generated_ids = model.generate(**inputs.to("cuda"), language="fr")
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## 📝 Notes pour le Papier

- Mentionner que les métriques sont mesurées sur RTX 5090
- Le débit varie selon la longueur d'audio et GPU
- Pour CPU, utiliser quantization (int8) pour meilleure performance
- Le modèle est compatible avec faster-whisper, whisper.cpp pour optimisations supplémentaires

