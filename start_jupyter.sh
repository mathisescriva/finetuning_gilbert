#!/bin/bash
# Script pour démarrer Jupyter sur RunPod

echo "🔧 Configuration Jupyter pour RunPod..."

# Installer Jupyter si nécessaire
pip install jupyter jupyterlab --quiet

# Créer répertoire pour Jupyter
mkdir -p /workspace/.jupyter

# Configuration Jupyter pour RunPod
cat > /workspace/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.base_url = '/'
c.ServerApp.allow_origin = '*'
EOF

echo "✅ Configuration créée"
echo "🚀 Démarrage de Jupyter Lab sur le port 8888..."
echo ""
echo "💡 Accédez via: https://m3djlqfzljissp-8888.proxy.runpod.net"
echo ""

# Démarrer Jupyter Lab en arrière-plan
nohup jupyter lab --config=/workspace/.jupyter/jupyter_lab_config.py --no-browser --allow-root > /workspace/jupyter.log 2>&1 &

echo "✅ Jupyter Lab démarré (PID: $!)"
echo "📋 Logs dans: /workspace/jupyter.log"
echo ""
echo "Pour vérifier: tail -f /workspace/jupyter.log"

