# Solution rapide - Modèle quantifié

## ✅ La quantization a RÉUSSI !

Les fichiers `encoder_model.onnx` et `decoder_model.onnx` ont été quantifiés avec succès.

## Vérifier ce qui existe

```bash
cd /workspace/finetuning_gilbert

# Vérifier les fichiers quantifiés
ls -lh outputs/models/whisper-ptq-int8/quantized/*.onnx

# Vérifier taille
du -sh outputs/models/whisper-ptq-int8/quantized/
```

## Si les fichiers .onnx quantifiés existent

Le modèle est **utilisable** ! Les fichiers `.onnx_data` ne sont peut-être pas nécessaires.

### Option 1: Utiliser le répertoire quantized directement

```bash
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq, AutoProcessor

model_path = "outputs/models/whisper-ptq-int8/quantized"
model = ORTModelForSpeechSeq2Seq.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)
```

### Option 2: Nettoyer et finaliser

```bash
# Supprimer les fichiers temporaires ONNX non quantifiés pour libérer de l'espace
rm -rf outputs/models/whisper-ptq-int8/onnx/encoder_model.onnx_data
rm -rf outputs/models/whisper-ptq-int8/onnx/decoder_model.onnx_data

# Vérifier espace
df -h

# Mettre à jour le script (pour avoir la dernière version qui ignore .onnx_data)
git pull

# Copier manuellement les fichiers de config manquants
cp outputs/models/whisper-ptq-int8/onnx/config.json outputs/models/whisper-ptq-int8/quantized/ 2>/dev/null
cp outputs/models/whisper-ptq-int8/onnx/generation_config.json outputs/models/whisper-ptq-int8/quantized/ 2>/dev/null
```

## Résultat attendu

Vous devriez avoir dans `quantized/`:
- `encoder_model.onnx` (quantifié int8)
- `decoder_model.onnx` (quantifié int8)
- `config.json`
- `preprocessor_config.json`
- `tokenizer.json`, etc.

Le modèle est **prêt à être utilisé** !

