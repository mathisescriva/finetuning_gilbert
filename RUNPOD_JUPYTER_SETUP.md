# Setup Jupyter sur RunPod

## Problème
Jupyter n'est pas démarré automatiquement sur le pod RunPod, d'où l'erreur 404.

## Solution : Démarrer Jupyter manuellement

### Option 1 : Via SSH (Recommandé)

1. **Se connecter en SSH** :
```bash
ssh -i ~/.ssh/id_ed25519 m3djlqfzljissp-64411a7a@ssh.runpod.io
```

2. **Sur RunPod, exécuter** :
```bash
cd /workspace/finetuning_gilbert

# Installer Jupyter
pip install jupyter jupyterlab

# Démarrer Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

3. **Laisser tourner** et accéder via : https://m3djlqfzljissp-8888.proxy.runpod.net

### Option 2 : Utiliser le script

1. **Uploader le script** (déjà dans votre projet)
2. **SSH sur RunPod** et exécuter :
```bash
cd /workspace/finetuning_gilbert
bash start_jupyter.sh
```

### Option 3 : Via Web Terminal de RunPod

1. Sur la page RunPod, activer "Enable web terminal"
2. Dans le terminal web, exécuter :
```bash
cd /workspace
pip install jupyter jupyterlab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

## Vérifier que ça fonctionne

Après avoir démarré Jupyter, vérifier :
```bash
# Vérifier que le processus tourne
ps aux | grep jupyter

# Vérifier les logs
tail -f /workspace/jupyter.log
```

Puis accéder à : https://m3djlqfzljissp-8888.proxy.runpod.net

## Astuce : Garder Jupyter actif

Pour que Jupyter continue même si vous fermez SSH :
```bash
# Utiliser screen ou tmux
screen -S jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# Détacher: Ctrl+A puis D

# Ou utiliser nohup
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root > jupyter.log 2>&1 &
```

