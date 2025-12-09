# Déploiement sur HuggingFace

## 🚀 Déployer le modèle ONNX Gilbert

### Prérequis

1. **Token HuggingFace** :
   ```bash
   # Option 1: Variable d'environnement
   export HUGGINGFACE_TOKEN="hf_xxxxx"
   
   # Option 2: Login HuggingFace CLI
   huggingface-cli login
   ```

2. **Installation dépendances** :
   ```bash
   pip install huggingface_hub
   ```

### Déploiement

```bash
# Déployer avec nom personnalisé
python scripts/deploy_to_hf.py \
  --repo_name "mathisescriva/gilbert-whisper-onnx" \
  --local_path "outputs/models/gilbert-whisper-ptq-int8/onnx"

# Si repo privé
python scripts/deploy_to_hf.py \
  --repo_name "mathisescriva/gilbert-whisper-onnx" \
  --private \
  --token "hf_xxxxx"
```

### Suggestions de noms

- `mathisescriva/gilbert-whisper-onnx`
- `mathisescriva/gilbert-stt-onnx`
- `mathisescriva/gilbert-whisper-fr-onnx`
- `mathisescriva/gilbert-whisper-optimized`

### Vérification

Après déploiement, vérifier sur :
- https://huggingface.co/votre-username/gilbert-whisper-onnx

### Utilisation depuis HuggingFace

```python
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor

model = ORTModelForSpeechSeq2Seq.from_pretrained("votre-username/gilbert-whisper-onnx")
processor = AutoProcessor.from_pretrained("votre-username/gilbert-whisper-onnx")
```

