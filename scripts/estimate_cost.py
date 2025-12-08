#!/usr/bin/env python3
"""
Script pour estimer le coût de transcription avec différents services
avant de lancer la transcription complète.
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset
import librosa
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))


def estimate_dataset_duration(dataset, audio_column: str = "audio", sample_size: int = None):
    """
    Estime la durée totale d'un dataset en analysant les fichiers audio.
    
    Args:
        dataset: Dataset HuggingFace
        audio_column: Nom de la colonne audio
        sample_size: Analyser seulement un échantillon (pour test rapide)
    
    Returns:
        Dict avec statistiques de durée
    """
    print(f"📊 Analyse de la durée du dataset...")
    print(f"   Total échantillons: {len(dataset)}")
    
    if sample_size and len(dataset) > sample_size:
        print(f"   ⚠️  Analyse sur échantillon de {sample_size} pour estimation rapide")
        dataset = dataset.select(range(sample_size))
        is_sample = True
    else:
        is_sample = False
    
    durations = []
    errors = 0
    
    for idx, example in enumerate(tqdm(dataset, desc="Analyse audio")):
        try:
            audio_data = example[audio_column]
            
            if audio_data is None:
                errors += 1
                continue
            
            # Calculer durée
            if isinstance(audio_data, dict):
                # Dataset HuggingFace avec audio chargé
                audio_array = audio_data.get("array")
                sampling_rate = audio_data.get("sampling_rate", 16000)
                
                if audio_array is not None:
                    duration = len(audio_array) / sampling_rate
                    durations.append(duration)
            
            elif isinstance(audio_data, str):
                # Chemin vers fichier
                try:
                    y, sr = librosa.load(audio_data, sr=None, duration=1.0)
                    # Charger juste pour avoir la durée réelle
                    import soundfile as sf
                    info = sf.info(audio_data)
                    duration = info.duration
                    durations.append(duration)
                except:
                    errors += 1
            
            else:
                # Array direct
                duration = len(audio_data) / 16000  # Assume 16kHz
                durations.append(duration)
        
        except Exception as e:
            errors += 1
            if idx < 5:  # Afficher seulement les premières erreurs
                print(f"   ⚠️  Erreur échantillon {idx}: {e}")
    
    if not durations:
        return {
            "total_duration_seconds": 0,
            "total_duration_minutes": 0,
            "total_duration_hours": 0,
            "avg_duration_seconds": 0,
            "errors": errors,
            "is_sample": is_sample,
        }
    
    total_duration_seconds = sum(durations)
    avg_duration = total_duration_seconds / len(durations)
    
    # Si échantillon, extrapoler
    if is_sample:
        original_size = len(dataset) if not sample_size else len(load_dataset(args.dataset_name, split=args.split)) if 'args' in locals() else len(dataset)
        extrapolated_seconds = total_duration_seconds * (original_size / sample_size)
        return {
            "total_duration_seconds": extrapolated_seconds,
            "total_duration_minutes": extrapolated_seconds / 60,
            "total_duration_hours": extrapolated_seconds / 3600,
            "avg_duration_seconds": avg_duration,
            "sample_size": sample_size,
            "original_size": original_size,
            "errors": errors,
            "is_sample": True,
        }
    
    return {
        "total_duration_seconds": total_duration_seconds,
        "total_duration_minutes": total_duration_seconds / 60,
        "total_duration_hours": total_duration_seconds / 3600,
        "avg_duration_seconds": avg_duration,
        "min_duration_seconds": min(durations),
        "max_duration_seconds": max(durations),
        "errors": errors,
        "is_sample": False,
    }


def calculate_costs(duration_hours: float):
    """
    Calcule les coûts pour différents services.
    
    Args:
        duration_hours: Durée totale en heures
    
    Returns:
        Dict avec coûts par service
    """
    duration_minutes = duration_hours * 60
    
    costs = {
        "assemblyai": {
            "price_per_minute": 0.0001,  # $0.0001/min
            "cost": duration_minutes * 0.0001,
            "free_credit": 50.0,  # $50 gratuit
            "cost_after_free": max(0, (duration_minutes * 0.0001) - 50.0),
            "free_hours": 50.0 / (60 * 0.0001),  # ~833 heures gratuites
        },
        "deepgram": {
            "price_per_minute": 0.0043 / 60,  # $0.0043/min
            "cost": duration_minutes * (0.0043 / 60),
            "free_credit": 0.0,
        },
        "azure": {
            "price_per_hour": 1.0,  # $1/hour
            "cost": duration_hours * 1.0,
            "free_credit": 0.0,
        },
        "google": {
            "price_per_15sec": 0.006,  # $0.006 per 15 sec
            "cost": (duration_hours * 3600 / 15) * 0.006,
            "free_credit": 0.0,
        },
        "whisper": {
            "price_per_hour": 0.0,  # Gratuit
            "cost": 0.0,
            "note": "Gratuit mais nécessite GPU/computation",
        },
    }
    
    return costs


def main():
    parser = argparse.ArgumentParser(description="Estimer le coût de transcription")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MEscriva/french-education-speech",
        help="Nom du dataset HuggingFace",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split à analyser",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Taille échantillon pour estimation rapide (None = tout analyser)",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="audio",
        help="Nom de la colonne audio",
    )
    
    args = parser.parse_args()
    
    # Charger dataset
    print(f"📥 Chargement du dataset {args.dataset_name}...")
    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return
    
    # Vérifier colonne audio
    if args.audio_column not in dataset.column_names:
        print(f"⚠️  Colonne '{args.audio_column}' non trouvée")
        print(f"   Colonnes disponibles: {dataset.column_names}")
        # Essayer de trouver automatiquement
        for col in ["audio", "path", "file"]:
            if col in dataset.column_names:
                args.audio_column = col
                print(f"   Utilisation de '{col}' à la place")
                break
        else:
            print("❌ Impossible de trouver colonne audio")
            return
    
    # Estimer durée
    stats = estimate_dataset_duration(
        dataset,
        audio_column=args.audio_column,
        sample_size=args.sample_size,
    )
    
    if stats["total_duration_hours"] == 0:
        print("❌ Impossible d'estimer la durée (aucun audio trouvé)")
        return
    
    # Afficher statistiques
    print(f"\n{'='*60}")
    print("📊 STATISTIQUES DU DATASET")
    print(f"{'='*60}")
    
    if stats.get("is_sample"):
        print(f"⚠️  Estimation basée sur échantillon de {stats['sample_size']} sur {stats['original_size']} total")
        print(f"   (Extrapolé à partir de l'échantillon)")
    
    print(f"\nDurée totale:")
    print(f"  {stats['total_duration_hours']:.2f} heures")
    print(f"  {stats['total_duration_minutes']:.2f} minutes")
    print(f"  {stats['total_duration_seconds']:.0f} secondes")
    
    if not stats.get("is_sample"):
        print(f"\nDurée moyenne par échantillon:")
        print(f"  {stats['avg_duration_seconds']:.2f} secondes")
        print(f"  Min: {stats['min_duration_seconds']:.2f}s")
        print(f"  Max: {stats['max_duration_seconds']:.2f}s")
    
    if stats['errors'] > 0:
        print(f"\n⚠️  {stats['errors']} échantillons avec erreurs")
    
    # Calculer coûts
    costs = calculate_costs(stats['total_duration_hours'])
    
    print(f"\n{'='*60}")
    print("💰 ESTIMATION DES COÛTS")
    print(f"{'='*60}")
    
    # AssemblyAI (priorité)
    print(f"\n🎯 AssemblyAI (RECOMMANDÉ):")
    ai_cost = costs["assemblyai"]
    print(f"  Prix: ${ai_cost['price_per_minute']*60:.4f} par heure")
    print(f"  Coût total: ${ai_cost['cost']:.2f}")
    print(f"  Crédit gratuit: ${ai_cost['free_credit']:.2f}")
    
    if ai_cost['cost'] <= ai_cost['free_credit']:
        print(f"  ✅ GRATUIT ! (dans le crédit gratuit)")
        print(f"  💡 Vous avez ${ai_cost['free_credit'] - ai_cost['cost']:.2f} de crédit restant")
    else:
        print(f"  💰 Coût réel: ${ai_cost['cost_after_free']:.2f}")
        print(f"  💡 Après crédit gratuit de ${ai_cost['free_credit']:.2f}")
    
    print(f"  📊 Crédit gratuit = ~{ai_cost['free_hours']:.0f} heures gratuites")
    
    # Autres services
    print(f"\n📋 Autres services:")
    print(f"  Deepgram:    ${costs['deepgram']['cost']:.2f}")
    print(f"  Azure:       ${costs['azure']['cost']:.2f}")
    print(f"  Google:      ${costs['google']['cost']:.2f}")
    print(f"  Whisper:     {costs['whisper']['cost']:.2f} (gratuit, mais GPU requis)")
    
    print(f"\n{'='*60}")
    print("💡 RECOMMANDATION")
    print(f"{'='*60}")
    
    if ai_cost['cost'] <= ai_cost['free_credit']:
        print(f"\n✅ Utilisez AssemblyAI - C'EST GRATUIT pour votre dataset !")
        print(f"   Vous avez assez de crédit gratuit pour tout transcrire.")
    elif ai_cost['cost_after_free'] < 10:
        print(f"\n✅ Utilisez AssemblyAI - Coût très faible (${ai_cost['cost_after_free']:.2f})")
        print(f"   Qualité excellente pour un petit prix.")
    elif stats['total_duration_hours'] < 100:
        print(f"\n✅ Utilisez AssemblyAI - Bon rapport qualité/prix")
        print(f"   Coût: ${ai_cost['cost_after_free']:.2f} pour {stats['total_duration_hours']:.1f}h")
    else:
        print(f"\n⚠️  Dataset volumineux ({stats['total_duration_hours']:.1f}h)")
        print(f"   Options:")
        print(f"   1. AssemblyAI: ${ai_cost['cost_after_free']:.2f} (qualité maximale)")
        print(f"   2. Whisper: Gratuit (qualité moindre mais acceptable)")
        print(f"   3. Mix: AssemblyAI pour échantillon + Whisper pour le reste")
    
    print(f"\n🚀 Pour lancer la transcription:")
    print(f"   python scripts/generate_transcripts_commercial.py \\")
    print(f"     --dataset_name {args.dataset_name} \\")
    print(f"     --service assemblyai \\")
    print(f"     --split {args.split}")


if __name__ == "__main__":
    main()

