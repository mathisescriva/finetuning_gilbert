#!/usr/bin/env python3
"""
Script pour télécharger et préparer des datasets publics français pour l'entraînement.
Utilise des datasets HuggingFace disponibles publiquement.
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import os


def download_common_voice_fr(output_dir: Path, splits: list = None):
    """
    Télécharge Mozilla Common Voice français.
    Dataset généraliste de parole en français, très utile pour fine-tuning.
    """
    if splits is None:
        splits = ["train", "validation", "test"]
    
    print("📥 Téléchargement de Mozilla Common Voice français...")
    
    try:
        dataset = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "fr",
            split="train+validation+test",
            trust_remote_code=True,
        )
        
        # Split en train/validation/test
        train_val_test = dataset.train_test_split(test_size=0.2, seed=42)
        val_test = train_val_test["test"].train_test_split(test_size=0.5, seed=42)
        
        dataset_dict = DatasetDict({
            "train": train_val_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        })
        
        # Filtrer colonnes et renommer
        def process_example(example):
            return {
                "audio": example["audio"],
                "text": example["sentence"],
            }
        
        for split in splits:
            if split in dataset_dict:
                dataset_dict[split] = dataset_dict[split].map(
                    process_example,
                    remove_columns=[col for col in dataset_dict[split].column_names if col not in ["audio", "text"]],
                )
        
        # Sauvegarder
        output_path = output_dir / "common_voice_fr"
        dataset_dict.save_to_disk(str(output_path))
        print(f"✅ Common Voice FR sauvegardé dans {output_path}")
        
        return dataset_dict
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement de Common Voice: {e}")
        print("💡 Tentative avec une version alternative...")
        try:
            # Alternative: common_voice_16_0
            dataset = load_dataset("mozilla-foundation/common_voice_16_0", "fr")
            dataset_dict = DatasetDict({
                "train": dataset["train"],
                "validation": dataset["validation"],
                "test": dataset["test"],
            })
            
            for split in splits:
                if split in dataset_dict:
                    dataset_dict[split] = dataset_dict[split].map(
                        lambda x: {"audio": x["audio"], "text": x["sentence"]},
                        remove_columns=[col for col in dataset_dict[split].column_names if col not in ["audio", "text"]],
                    )
            
            output_path = output_dir / "common_voice_fr"
            dataset_dict.save_to_disk(str(output_path))
            print(f"✅ Common Voice FR (v16) sauvegardé dans {output_path}")
            return dataset_dict
        except Exception as e2:
            print(f"❌ Erreur avec version alternative: {e2}")
            return None


def download_mls_french(output_dir: Path):
    """
    Télécharge Multilingual LibriSpeech français.
    Dataset de haute qualité avec lecture de livres en français.
    """
    print("📥 Téléchargement de Multilingual LibriSpeech français...")
    
    try:
        # MLS français est souvent disponible
        dataset = load_dataset("facebook/multilingual_librispeech", "french", split="train")
        
        # Créer splits
        dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
        val_test = dataset_split["test"].train_test_split(test_size=0.5, seed=42)
        
        dataset_dict = DatasetDict({
            "train": dataset_split["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        })
        
        # Processer pour format uniforme
        def process_example(example):
            return {
                "audio": example["audio"],
                "text": example["text"],
            }
        
        for split in ["train", "validation", "test"]:
            dataset_dict[split] = dataset_dict[split].map(
                process_example,
                remove_columns=[col for col in dataset_dict[split].column_names if col not in ["audio", "text"]],
            )
        
        output_path = output_dir / "mls_french"
        dataset_dict.save_to_disk(str(output_path))
        print(f"✅ MLS French sauvegardé dans {output_path}")
        
        return dataset_dict
        
    except Exception as e:
        print(f"⚠️  MLS French non disponible ou erreur: {e}")
        return None


def download_voxpopuli_fr(output_dir: Path):
    """
    Télécharge VoxPopuli français.
    Données parlementaires européennes en français - plus proche du style réunions.
    """
    print("📥 Téléchargement de VoxPopuli français...")
    
    try:
        # VoxPopuli a des données parlementaires
        dataset = load_dataset("facebook/voxpopuli", "fr", split="train")
        
        # Limiter la taille (très gros dataset)
        if len(dataset) > 10000:
            dataset = dataset.select(range(10000))
        
        dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
        val_test = dataset_split["test"].train_test_split(test_size=0.5, seed=42)
        
        dataset_dict = DatasetDict({
            "train": dataset_split["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        })
        
        def process_example(example):
            return {
                "audio": example["audio"],
                "text": example["raw_text"] if "raw_text" in example else example.get("text", ""),
            }
        
        for split in ["train", "validation", "test"]:
            dataset_dict[split] = dataset_dict[split].map(
                process_example,
                remove_columns=[col for col in dataset_dict[split].column_names if col not in ["audio", "text"]],
            )
        
        output_path = output_dir / "voxpopuli_fr"
        dataset_dict.save_to_disk(str(output_path))
        print(f"✅ VoxPopuli FR sauvegardé dans {output_path}")
        
        return dataset_dict
        
    except Exception as e:
        print(f"⚠️  VoxPopuli FR non disponible ou erreur: {e}")
        return None


def create_combined_dataset(datasets_dict: dict, output_dir: Path, max_samples_per_dataset: int = None):
    """
    Combine plusieurs datasets en un seul.
    """
    print("\n🔄 Combinaison des datasets...")
    
    combined = {"train": [], "validation": [], "test": []}
    
    for dataset_name, dataset_dict in datasets_dict.items():
        if dataset_dict is None:
            continue
        
        print(f"  Ajout de {dataset_name}...")
        for split in ["train", "validation", "test"]:
            if split in dataset_dict:
                samples = list(dataset_dict[split])
                
                # Limiter si demandé
                if max_samples_per_dataset and len(samples) > max_samples_per_dataset:
                    import random
                    random.seed(42)
                    samples = random.sample(samples, max_samples_per_dataset)
                    print(f"    {split}: {len(samples)} échantillons (limité)")
                else:
                    print(f"    {split}: {len(samples)} échantillons")
                
                combined[split].extend(samples)
    
    # Créer DatasetDict
    from datasets import DatasetDict, Dataset
    
    combined_dict = DatasetDict({
        split: Dataset.from_list(samples) if samples else Dataset.from_dict({"audio": [], "text": []})
        for split, samples in combined.items()
    })
    
    # Statistiques
    print("\n📊 Statistiques du dataset combiné:")
    for split in ["train", "validation", "test"]:
        print(f"  {split}: {len(combined_dict[split])} échantillons")
    
    # Sauvegarder
    output_path = output_dir / "combined_french_asr"
    combined_dict.save_to_disk(str(output_path))
    print(f"\n✅ Dataset combiné sauvegardé dans {output_path}")
    
    return combined_dict


def export_to_json_format(dataset_dict: DatasetDict, output_dir: Path, max_samples: int = None):
    """
    Exporte le dataset au format JSON pour utilisation avec nos scripts.
    Note: Les fichiers audio restent dans le cache HuggingFace.
    """
    print("\n📝 Export en format JSON...")
    
    for split in ["train", "validation", "test"]:
        if split not in dataset_dict:
            continue
        
        samples = []
        dataset = dataset_dict[split]
        
        # Limiter si demandé (pour test rapide)
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        print(f"  Export {split}...")
        for i, example in enumerate(tqdm(dataset, desc=f"  {split}")):
            # Pour les datasets HuggingFace, l'audio est chargé dynamiquement
            # On note juste le chemin ou l'info
            sample = {
                "audio": {
                    "path": example["audio"]["path"] if "path" in example["audio"] else None,
                    "array": None,  # Pas stocké en JSON
                },
                "text": example["text"],
            }
            
            # Si on a un chemin, on le garde, sinon on note que c'est dans le dataset
            if sample["audio"]["path"]:
                samples.append({
                    "audio": sample["audio"]["path"],
                    "text": sample["text"],
                })
            else:
                # Fallback: noter l'index dans le dataset
                samples.append({
                    "audio": f"dataset_index:{i}",  # Nécessitera chargement depuis dataset
                    "text": sample["text"],
                })
        
        # Sauvegarder JSON
        json_path = output_dir / f"{split}_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"  ✅ {split}: {len(samples)} échantillons → {json_path}")
    
    print("\n💡 Note: Les datasets HuggingFace chargent l'audio automatiquement.")
    print("   Utilisez directement le dataset HuggingFace avec --train_data <dataset_path>")


def main():
    parser = argparse.ArgumentParser(description="Télécharger datasets publics français")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Répertoire de sortie",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["common_voice", "mls", "voxpopuli", "all"],
        default=["common_voice"],
        help="Datasets à télécharger",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combiner tous les datasets en un",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Nombre maximum d'échantillons par dataset (pour test rapide)",
    )
    parser.add_argument(
        "--export_json",
        action="store_true",
        help="Exporter aussi au format JSON",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["common_voice", "mls", "voxpopuli"]
    
    downloaded_datasets = {}
    
    # Télécharger datasets
    if "common_voice" in datasets_to_download:
        downloaded_datasets["common_voice_fr"] = download_common_voice_fr(output_dir)
    
    if "mls" in datasets_to_download:
        downloaded_datasets["mls_french"] = download_mls_french(output_dir)
    
    if "voxpopuli" in datasets_to_download:
        downloaded_datasets["voxpopuli_fr"] = download_voxpopuli_fr(output_dir)
    
    # Filtrer None (datasets qui ont échoué)
    downloaded_datasets = {k: v for k, v in downloaded_datasets.items() if v is not None}
    
    if not downloaded_datasets:
        print("\n❌ Aucun dataset téléchargé avec succès.")
        return
    
    # Combiner si demandé
    if args.combine:
        combined = create_combined_dataset(downloaded_datasets, output_dir, args.max_samples)
        downloaded_datasets["combined"] = combined
    
    # Exporter JSON si demandé
    if args.export_json:
        for name, dataset_dict in downloaded_datasets.items():
            if dataset_dict is not None:
                dataset_dir = output_dir / name
                export_to_json_format(dataset_dict, dataset_dir, args.max_samples)
    
    print("\n" + "="*60)
    print("✅ TÉLÉCHARGEMENT TERMINÉ")
    print("="*60)
    print("\n📚 Datasets disponibles:")
    for name, dataset_dict in downloaded_datasets.items():
        if dataset_dict is not None:
            print(f"\n  {name}:")
            print(f"    Chemin: {output_dir / name}")
            for split in ["train", "validation", "test"]:
                if split in dataset_dict:
                    print(f"    {split}: {len(dataset_dict[split])} échantillons")
    
    print("\n💡 Pour utiliser avec fine-tuning:")
    print(f"   python scripts/fine_tune_meetings.py \\")
    print(f"     --train_data {output_dir}/common_voice_fr \\")
    print(f"     --eval_data {output_dir}/common_voice_fr")
    print("\n   (Les datasets HuggingFace sont chargés automatiquement)")


if __name__ == "__main__":
    main()

