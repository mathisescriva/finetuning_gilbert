#!/usr/bin/env python3
"""
Génère un tableau comparatif formaté pour publication (LaTeX/Markdown)
"""

import json
import argparse
from pathlib import Path


def generate_latex_table(results_path):
    """Génère un tableau LaTeX"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    model_size = results.get("model_size_gb", 0)
    baseline_size = results.get("baseline_size_gb", 0)
    avg_wer = results.get("average_wer", 0)
    avg_cer = results.get("average_cer", 0)
    speedup = results.get("speedup_vs_baseline", 1.0)
    size_reduction = results.get("size_reduction_percent", 0)
    wer_degradation = results.get("wer_degradation_vs_baseline", 0)
    
    # Métriques par dataset
    quality = results.get("quality_metrics", {})
    
    table = """\\begin{table*}[t]
\\centering
\\caption{Comparaison des modèles Whisper pour transcription française}
\\label{tab:model_comparison}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Modèle} & \\textbf{Taille} & \\textbf{WER} & \\textbf{CER} & \\textbf{Vitesse} & \\textbf{Accél.} & \\textbf{VRAM} \\\\
\\midrule
"""
    
    # Baseline
    baseline_wer = results.get("baseline_quality", {}).get("wer", 0)
    baseline_speed = results.get("baseline_inference", {}).get("mean_time", 0)
    table += f"Whisper Large-v3 & {baseline_size:.2f} GB & {baseline_wer:.2f}\\% & - & {baseline_speed:.3f}s & 1.0x & - \\\\\n"
    
    # Notre modèle
    inference = results.get("inference_benchmark", {})
    speed = inference.get("mean_time", 0)
    vram = inference.get("peak_memory_gb", 0)
    table += f"Distil-French v0.2 & {model_size:.2f} GB & {avg_wer:.2f}\\% & {avg_cer:.2f}\\% & {speed:.3f}s & {speedup:.2f}x & {vram:.2f} GB \\\\\n"
    
    table += """\\bottomrule
\\end{tabular}
\\end{table*}
"""
    
    return table


def generate_markdown_table(results_path):
    """Génère un tableau Markdown"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    model_size = results.get("model_size_gb", 0)
    baseline_size = results.get("baseline_size_gb", 0)
    avg_wer = results.get("average_wer", 0)
    avg_cer = results.get("average_cer", 0)
    speedup = results.get("speedup_vs_baseline", 1.0)
    size_reduction = results.get("size_reduction_percent", 0)
    wer_degradation = results.get("wer_degradation_vs_baseline", 0)
    
    inference = results.get("inference_benchmark", {})
    speed = inference.get("mean_time", 0)
    vram = inference.get("peak_memory_gb", 0)
    
    table = """## Tableau comparatif des modèles

| Modèle | Taille | Paramètres | WER | CER | Vitesse | Accélération | VRAM |
|--------|--------|------------|-----|-----|---------|--------------|------|
"""
    
    baseline_params = results.get("baseline_num_parameters", 0) / 1e6
    baseline_wer = results.get("baseline_quality", {}).get("wer", 0)
    baseline_speed = results.get("baseline_inference", {}).get("mean_time", 0)
    model_params = results.get("num_parameters", 0) / 1e6
    
    table += f"| Whisper Large-v3 | {baseline_size:.2f} GB | {baseline_params:.1f}M | {baseline_wer:.2f}% | - | {baseline_speed:.3f}s | 1.0x | - |\n"
    table += f"| **Distil-French v0.2** | **{model_size:.2f} GB** | **{model_params:.1f}M** | **{avg_wer:.2f}%** | **{avg_cer:.2f}%** | **{speed:.3f}s** | **{speedup:.2f}x** | **{vram:.2f} GB** |\n"
    
    table += f"\n### Métriques détaillées\n\n"
    table += f"- **Réduction de taille**: {size_reduction:.1f}%\n"
    table += f"- **Accélération**: {speedup:.2f}x plus rapide\n"
    table += f"- **Dégradation WER**: {wer_degradation:+.2f}% (vs baseline)\n"
    
    # Métriques par dataset
    quality = results.get("quality_metrics", {})
    if quality:
        table += f"\n### Performance par dataset\n\n"
        table += "| Dataset | WER | CER | Échantillons |\n"
        table += "|---------|-----|-----|--------------|\n"
        for dataset_name, metrics in quality.items():
            if "wer" in metrics:
                table += f"| {dataset_name} | {metrics['wer']:.2f}% | {metrics['cer']:.2f}% | {metrics['num_samples']} |\n"
    
    return table


def main():
    parser = argparse.ArgumentParser(description="Génère tableaux pour publication")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Fichier JSON de résultats",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["latex", "markdown", "both"],
        default="both",
        help="Format de sortie",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Fichier de sortie (si non spécifié, print)",
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"❌ Fichier non trouvé: {results_path}")
        return
    
    output_path = Path(args.output) if args.output else None
    
    if args.format in ["latex", "both"]:
        latex_table = generate_latex_table(results_path)
        if output_path:
            latex_file = output_path.parent / f"{output_path.stem}_latex.txt"
            latex_file.write_text(latex_table)
            print(f"✅ Tableau LaTeX sauvegardé: {latex_file}")
        else:
            print("="*80)
            print("LATEX TABLE:")
            print("="*80)
            print(latex_table)
    
    if args.format in ["markdown", "both"]:
        md_table = generate_markdown_table(results_path)
        if output_path:
            md_file = output_path.parent / f"{output_path.stem}_table.md" if output_path.suffix == ".json" else Path(args.output)
            md_file.write_text(md_table)
            print(f"✅ Tableau Markdown sauvegardé: {md_file}")
        else:
            print("="*80)
            print("MARKDOWN TABLE:")
            print("="*80)
            print(md_table)


if __name__ == "__main__":
    main()

