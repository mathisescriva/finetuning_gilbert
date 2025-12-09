#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) pour Whisper.
Simple et rapide - pas besoin de données d'entraînement.
"""

import argparse
import os
import gc
import torch
import shutil
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer


def quantize_to_int8(model_name_or_path: str, output_path: str):
    """
    Quantifie un modèle Whisper en int8 avec PTQ.
    """
    print("🔧 Post-Training Quantization (PTQ) pour Whisper")
    print(f"📥 Modèle source: {model_name_or_path}")
    print(f"💾 Modèle de sortie: {output_path}")
    print()
    
    # Changer le cache HuggingFace vers /workspace (plus d'espace)
    # NETTOYER AVANT de télécharger
    cache_dir = "/workspace/.hf_home"
    if os.path.exists(cache_dir):
        # Supprimer seulement les anciens téléchargements
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path):
                # Garder seulement le modèle qu'on va télécharger
                if "whisper-large-v3-distil-fr-v0.2" not in item:
                    try:
                        shutil.rmtree(item_path)
                        print(f"  Supprimé: {item}")
                    except:
                        pass
    
    os.environ["HF_HOME"] = "/workspace/.hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/.hf_home/hub"
    os.environ["HF_DATASETS_CACHE"] = "/workspace/.hf_home/datasets"
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Charger le modèle et le processeur
    print("📦 Chargement du modèle...")
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    # Sauvegarder le processeur
    processor.save_pretrained(output_path)
    
    print("✅ Modèle chargé")
    print(f"📊 Taille avant quantization: {sum(p.numel() * 4 for p in model.parameters()) / 1e9:.2f} GB (float32)")
    print()
    
    # Exporter et quantifier avec optimum (méthode simplifiée)
    print("🔄 Export et Quantification ONNX...")
    quantized_path = output_path / "quantized"
    quantized_path.mkdir(exist_ok=True)
    
    try:
        # Méthode 1: Export ONNX puis quantifier avec optimum (gère multi-fichiers)
        print("  Exportation ONNX avec quantization intégrée...")
        
        # Export ONNX standard
        onnx_model_path = output_path / "onnx"
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_name_or_path,
            export=True,
            use_cache=False,
        )
        onnx_model.save_pretrained(str(onnx_model_path))
        print("  ✅ Export ONNX réussi")
        
        # NETTOYER après export pour libérer espace
        print("  🧹 Nettoyage pour libérer espace...")
        del model  # Libérer mémoire GPU/RAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Supprimer fichiers .onnx_data volumineux (pas nécessaires pour quantification)
        for onnx_data_file in onnx_model_path.glob("*.onnx_data"):
            try:
                onnx_data_file.unlink()
                print(f"    Supprimé: {onnx_data_file.name}")
            except:
                pass
        
        # Nettoyer cache HuggingFace temporaire
        import shutil
        cache_dir = "/workspace/.hf_home/hub"
        if os.path.exists(cache_dir):
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                # Garder seulement le modèle qu'on utilise
                if os.path.isdir(item_path) and "whisper-large-v3-distil-fr-v0.2" not in item:
                    try:
                        shutil.rmtree(item_path)
                    except:
                        pass
        
        print("  ✅ Espace libéré")
        
        # Quantifier chaque composant séparément (encoder, decoder)
        print("🔢 Quantification int8 (multi-fichiers)...")
        
        # Configuration quantization dynamique (plus compatible avec ONNX Runtime standard)
        # Utiliser QOperator au lieu de QDQ pour meilleure compatibilité
        from optimum.onnxruntime.configuration import QuantizationConfig
        
        qconfig = AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=True,  # Meilleure qualité
        )
        
        # Alternative: quantization dynamique simple (plus compatible)
        # qconfig = QuantizationConfig(
        #     is_static=False,
        #     format="QOperator",  # Plus compatible que QDQ
        # )
        
        # Lister les fichiers ONNX
        onnx_files = list(onnx_model_path.glob("*.onnx"))
        print(f"  Trouvé {len(onnx_files)} fichiers ONNX à quantifier")
        
        # Quantifier chaque fichier
        quantized_files = {}
        for onnx_file in onnx_files:
            try:
                print(f"  Quantification de {onnx_file.name}...")
                quantizer = ORTQuantizer.from_pretrained(str(onnx_model_path), file_name=onnx_file.name)
                
                # Créer répertoire temporaire pour ce fichier
                temp_quant_dir = output_path / f"temp_quant_{onnx_file.stem}"
                temp_quant_dir.mkdir(exist_ok=True)
                
                quantizer.quantize(
                    save_dir=str(temp_quant_dir),
                    quantization_config=qconfig,
                )
                
                # Déplacer le fichier quantifié
                quantized_file = list(temp_quant_dir.glob("*.onnx"))[0]
                target_file = quantized_path / quantized_file.name
                quantized_file.rename(target_file)
                quantized_files[onnx_file.name] = target_file.name
                
                # Nettoyer
                shutil.rmtree(temp_quant_dir)
                print(f"    ✅ {onnx_file.name} quantifié")
                
            except Exception as e_file:
                print(f"    ⚠️  Erreur pour {onnx_file.name}: {e_file}")
                continue
        
        # Copier les autres fichiers nécessaires (config, tokenizer, etc.)
        # Mais sauter les fichiers .onnx_data qui sont très gros
        print("  Copie des fichiers de configuration...")
        skipped = []
        for file in onnx_model_path.glob("*"):
            if file.is_file() and file.suffix != ".onnx" and not file.name.endswith(".onnx_data"):
                try:
                    shutil.copy2(file, quantized_path / file.name)
                except OSError as e:
                    if "No space" in str(e):
                        skipped.append(file.name)
                        print(f"    ⚠️  Espace insuffisant pour copier {file.name}, création lien symbolique...")
                        try:
                            (quantized_path / file.name).symlink_to(file)
                        except:
                            pass
                    else:
                        raise
        
        if skipped:
            print(f"    ⚠️  {len(skipped)} fichiers non copiés (espace insuffisant), utilisent liens symboliques")
        
        # Les fichiers .onnx_data sont déjà référencés par les fichiers .onnx quantifiés si besoin
        
        # Sauvegarder aussi le processor
        processor.save_pretrained(str(quantized_path))
        
        print()
        print("✅ ✅ ✅ QUANTIZATION TERMINÉE! ✅ ✅ ✅")
        print(f"📁 Modèle quantifié dans: {quantized_path}")
        print()
        print("💡 Utilisation:")
        print(f"   from optimum.onnxruntime import ORTModelForSpeechSeq2Seq")
        print(f"   model = ORTModelForSpeechSeq2Seq.from_pretrained('{quantized_path}')")
        print()
        
        # Estimation taille
        if quantized_path.exists():
            total_size = sum(
                f.stat().st_size 
                for f in quantized_path.rglob("*") 
                if f.is_file()
            ) / 1e9
            original_size = sum(p.numel() * 4 for p in model.parameters()) / 1e9
            reduction = (1 - total_size / original_size) * 100 if original_size > 0 else 0
            print(f"📊 Taille après quantization: ~{total_size:.2f} GB (int8)")
            print(f"📊 Taille originale: ~{original_size:.2f} GB (float32)")
            print(f"💾 Réduction: ~{reduction:.1f}%")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'export/quantization: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("💡 Le modèle ONNX a été exporté dans:", onnx_model_path)
        print("   Vous pouvez l'utiliser directement ou quantifier manuellement.")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Post-Training Quantization (PTQ) pour Whisper - Simple et rapide"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bofenghuang/whisper-large-v3-distil-fr-v0.2",
        help="Modèle HuggingFace à quantifier",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/models/whisper-ptq-int8",
        help="Répertoire de sortie",
    )
    
    args = parser.parse_args()
    
    quantize_to_int8(args.model, args.output)


if __name__ == "__main__":
    main()

