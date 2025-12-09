#!/bin/bash
# Script pour libérer de l'espace disque sur Vast.ai

echo "🧹 Libération d'espace disque..."
echo ""

# 1. Nettoyer cache HuggingFace (garde seulement le modèle chargé)
echo "📦 Nettoyage cache HuggingFace..."
rm -rf /workspace/.hf_home/hub/datasets--facebook--multilingual_librispeech 2>/dev/null || true
rm -rf /workspace/.hf_home/hub/datasets--mozilla-foundation--common_voice* 2>/dev/null || true

# Garder seulement les modèles
echo "   ✅ Cache datasets nettoyé"

# 2. Nettoyer cache pip
echo "📦 Nettoyage cache pip..."
pip cache purge 2>/dev/null || true
echo "   ✅ Cache pip nettoyé"

# 3. Nettoyer fichiers temporaires
echo "📦 Nettoyage fichiers temporaires..."
rm -rf /tmp/* 2>/dev/null || true
rm -rf /workspace/tmp/* 2>/dev/null || true
echo "   ✅ Fichiers temporaires nettoyés"

# 4. Afficher espace disponible
echo ""
echo "💾 Espace disque disponible:"
df -h /workspace | tail -1

echo ""
echo "✅ Nettoyage terminé !"
echo ""
echo "💡 Pour utiliser streaming (recommandé avec peu d'espace):"
echo "   Les scripts ont été mis à jour pour utiliser streaming automatiquement"

