#!/usr/bin/env python3
"""
Script de test rapide pour vérifier que le téléchargement de datasets fonctionne.
Télécharge un petit échantillon pour test.
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.download_datasets import download_common_voice_fr

def main():
    """Test rapide du téléchargement."""
    print("🧪 Test de téléchargement de Common Voice français...")
    print("   (échantillon limité pour test rapide)\n")
    
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Télécharger avec échantillon limité
        dataset = download_common_voice_fr(output_dir, splits=["train"])
        
        if dataset is not None:
            print(f"\n✅ Succès ! Dataset téléchargé dans {output_dir}/common_voice_fr")
            print(f"   Train: {len(dataset['train'])} échantillons")
            
            # Afficher un exemple
            example = dataset["train"][0]
            print(f"\n📝 Exemple:")
            print(f"   Texte: {example['text'][:100]}...")
            print(f"   Audio: {example['audio']}")
            
            print("\n💡 Pour télécharger le dataset complet:")
            print("   python scripts/download_datasets.py --datasets common_voice")
        else:
            print("\n❌ Échec du téléchargement")
            print("   Vérifiez votre connexion internet et les dépendances")
            
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

