#!/usr/bin/env python3
"""
Analyze deconvoluted genes output and visualize distributions.

Shows the best annotation type for each perturbation and graphs the distribution
of in_hop1 vs not_in_hop1 genes across all perturbations.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_deconvoluted_data(json_path: Path) -> Dict:
    """
    Load deconvoluted genes JSON file.
    
    Args:
        json_path: Path to deconvoluted_genes.json file
        
    Returns:
        Dictionary with deconvoluted data
    """
    logging.info(f"Loading deconvoluted data from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    logging.info(f"Loaded data for {len(data)} target genes")
    return data


def determine_best_annotation(
    target_data: Dict[str, Dict[str, List[str]]],
    annotation_types: List[str] = ["reactome", "bp", "cc", "mf"]
) -> Tuple[str, Dict]:
    """
    Determine the best annotation type for a target based on various metrics.
    
    Best is determined by:
    1. Highest number of deconvoluted genes (not in hop 1)
    2. Highest ratio of deconvoluted to total DE genes
    3. Highest total number of DE genes
    
    Args:
        target_data: Data for a single target gene
        annotation_types: List of annotation types to consider
        
    Returns:
        Tuple of (best_annotation_type, metrics_dict)
    """
    metrics = {}
    
    deconvoluted_all = set(target_data.get("deconvoluted", []))
    total_de = len(deconvoluted_all)
    
    for ann_type in annotation_types:
        if ann_type not in target_data:
            continue
        
        ann_data = target_data[ann_type]
        in_hop1 = set(ann_data.get("in_hop1", []))
        not_in_hop1 = set(ann_data.get("not_in_hop1", []))
        
        total_ann_de = len(in_hop1 | not_in_hop1)
        deconvoluted_ann = len(not_in_hop1)
        
        if total_ann_de > 0:
            ratio = deconvoluted_ann / total_ann_de
        else:
            ratio = 0.0
        
        metrics[ann_type] = {
            "total_de": total_ann_de,
            "in_hop1": len(in_hop1),
            "not_in_hop1": deconvoluted_ann,
            "deconvoluted_ratio": ratio,
            "deconvoluted_count": deconvoluted_ann
        }
    
    if not metrics:
        return None, {}
    
    # Determine best: prioritize deconvoluted count, then ratio, then total DE
    best_ann = max(
        metrics.keys(),
        key=lambda x: (
            metrics[x]["deconvoluted_count"],
            metrics[x]["deconvoluted_ratio"],
            metrics[x]["total_de"]
        )
    )
    
    return best_ann, metrics


def analyze_deconvoluted_data(data: Dict) -> pd.DataFrame:
    """
    Analyze deconvoluted data and create summary DataFrame.
    
    Args:
        data: Deconvoluted genes data dictionary
        
    Returns:
        DataFrame with analysis results
    """
    annotation_types = ["reactome", "bp", "cc", "mf"]
    rows = []
    
    for target_gene, target_data in data.items():
        best_ann, metrics = determine_best_annotation(target_data, annotation_types)
        
        if best_ann is None:
            continue
        
        best_metrics = metrics[best_ann]
        
        row = {
            "target": target_gene,
            "best_annotation": best_ann,
            "total_deconvoluted": len(target_data.get("deconvoluted", [])),
            "best_total_de": best_metrics["total_de"],
            "best_in_hop1": best_metrics["in_hop1"],
            "best_not_in_hop1": best_metrics["not_in_hop1"],
            "best_deconvoluted_ratio": best_metrics["deconvoluted_ratio"],
        }
        
        # Add metrics for all annotation types
        for ann_type in annotation_types:
            if ann_type in metrics:
                row[f"{ann_type}_total_de"] = metrics[ann_type]["total_de"]
                row[f"{ann_type}_in_hop1"] = metrics[ann_type]["in_hop1"]
                row[f"{ann_type}_not_in_hop1"] = metrics[ann_type]["not_in_hop1"]
                row[f"{ann_type}_ratio"] = metrics[ann_type]["deconvoluted_ratio"]
            else:
                row[f"{ann_type}_total_de"] = 0
                row[f"{ann_type}_in_hop1"] = 0
                row[f"{ann_type}_not_in_hop1"] = 0
                row[f"{ann_type}_ratio"] = 0.0
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def create_distribution_plots(df: pd.DataFrame, output_dir: Path):
    """
    Create visualization plots for the distribution analysis.
    
    Args:
        df: Analysis DataFrame
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Best annotation type distribution
    plt.figure(figsize=(10, 6))
    best_ann_counts = df['best_annotation'].value_counts()
    colors = sns.color_palette("husl", len(best_ann_counts))
    plt.bar(best_ann_counts.index, best_ann_counts.values, color=colors)
    plt.xlabel('Annotation Type', fontsize=12)
    plt.ylabel('Number of Perturbations', fontsize=12)
    plt.title('Distribution of Best Annotation Type Across Perturbations', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    for i, (ann, count) in enumerate(best_ann_counts.items()):
        plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'best_annotation_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved best annotation distribution plot")
    
    # 2. In hop1 vs Not in hop1 distribution (all perturbations) - Line plot with area fill
    # Use much larger figure size for readability
    fig, axes = plt.subplots(2, 2, figsize=(50, 14))
    axes = axes.flatten()
    
    annotation_types = ["reactome", "bp", "cc", "mf"]
    
    for idx, ann_type in enumerate(annotation_types):
        ax = axes[idx]
        
        in_hop1_col = f"{ann_type}_in_hop1"
        not_in_hop1_col = f"{ann_type}_not_in_hop1"
        
        # Filter to perturbations with DE genes for this annotation
        ann_df = df[df[f"{ann_type}_total_de"] > 0].copy()
        
        if len(ann_df) == 0:
            ax.text(0.5, 0.5, f'No data for {ann_type}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{ann_type.upper()}', fontsize=12, fontweight='bold')
            continue
        
        # Sort by total DE and use all perturbations
        ann_df_sorted = ann_df.sort_values(f"{ann_type}_total_de", ascending=False)
        
        # Create x positions for all perturbations
        x_pos = np.arange(len(ann_df_sorted))
        
        in_hop1_vals = ann_df_sorted[in_hop1_col].values
        not_in_hop1_vals = ann_df_sorted[not_in_hop1_col].values
        total_vals = in_hop1_vals + not_in_hop1_vals
        
        # Create area plot (stacked) with line overlay for better visibility
        ax.fill_between(x_pos, 0, in_hop1_vals, label='In Hop 1', color='#3498db', alpha=0.6, linewidth=0)
        ax.fill_between(x_pos, in_hop1_vals, total_vals, label='Not in Hop 1', color='#e74c3c', alpha=0.6, linewidth=0)
        
        # Add line plots on top for better visibility
        ax.plot(x_pos, in_hop1_vals, color='#2980b9', linewidth=0.5, alpha=0.8)
        ax.plot(x_pos, total_vals, color='#c0392b', linewidth=0.5, alpha=0.8)
        
        ax.set_xlabel('Perturbation Index (sorted by total DE)', fontsize=12)
        ax.set_ylabel('Number of Genes', fontsize=12)
        ax.set_title(f'{ann_type.upper()}\n(n={len(ann_df)} perturbations)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # Auto-scale y-axis to show full range
        max_val = max(total_vals.max(), 1)  # Ensure at least 1 for visibility
        ax.set_ylim(0, max_val * 1.05)
        
        # Set x-axis limits
        ax.set_xlim(0, len(ann_df_sorted) - 1)
        
        # Add some x-axis ticks for reference (every 500 perturbations)
        if len(ann_df_sorted) > 100:
            tick_positions = np.arange(0, len(ann_df_sorted), max(500, len(ann_df_sorted) // 5))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f'{int(x)}' for x in tick_positions], fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'in_hop1_vs_not_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved in_hop1 vs not_in_hop1 distribution plot (all {len(df)} perturbations)")
    
    # 2b. Create smoothed version for better readability
    fig, axes = plt.subplots(2, 2, figsize=(30, 10))
    axes = axes.flatten()
    
    for idx, ann_type in enumerate(annotation_types):
        ax = axes[idx]
        
        in_hop1_col = f"{ann_type}_in_hop1"
        not_in_hop1_col = f"{ann_type}_not_in_hop1"
        
        ann_df = df[df[f"{ann_type}_total_de"] > 0].copy()
        
        if len(ann_df) == 0:
            ax.text(0.5, 0.5, f'No data for {ann_type}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{ann_type.upper()}', fontsize=12, fontweight='bold')
            continue
        
        ann_df_sorted = ann_df.sort_values(f"{ann_type}_total_de", ascending=False)
        x_pos = np.arange(len(ann_df_sorted))
        
        in_hop1_vals = ann_df_sorted[in_hop1_col].values
        not_in_hop1_vals = ann_df_sorted[not_in_hop1_col].values
        total_vals = in_hop1_vals + not_in_hop1_vals
        
        # Apply rolling average for smoothing (window size ~5% of data or min 50)
        window_size = max(50, len(ann_df_sorted) // 20)
        if window_size % 2 == 0:
            window_size += 1  # Make odd for better centering
        
        # Convert to Series for rolling
        in_hop1_series = pd.Series(in_hop1_vals)
        total_series = pd.Series(total_vals)
        
        in_hop1_smooth = in_hop1_series.rolling(window=window_size, center=True).mean()
        total_smooth = total_series.rolling(window=window_size, center=True).mean()
        not_in_hop1_smooth = total_smooth - in_hop1_smooth
        
        # Create smoothed area plot
        ax.fill_between(x_pos, 0, in_hop1_smooth, label='In Hop 1 (smoothed)', color='#3498db', alpha=0.7)
        ax.fill_between(x_pos, in_hop1_smooth, total_smooth, label='Not in Hop 1 (smoothed)', color='#e74c3c', alpha=0.7)
        
        # Add smoothed lines
        ax.plot(x_pos, in_hop1_smooth, color='#2980b9', linewidth=1.5, alpha=0.9)
        ax.plot(x_pos, total_smooth, color='#c0392b', linewidth=1.5, alpha=0.9)
        
        ax.set_xlabel('Perturbation Index (sorted by total DE)', fontsize=11)
        ax.set_ylabel('Number of Genes (smoothed)', fontsize=11)
        ax.set_title(f'{ann_type.upper()} - Smoothed\n(n={len(ann_df)} perturbations, window={window_size})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        max_val = max(total_smooth.max(), 1)
        ax.set_ylim(0, max_val * 1.05)
        ax.set_xlim(0, len(ann_df_sorted) - 1)
        
        # Add x-axis ticks
        if len(ann_df_sorted) > 100:
            tick_positions = np.arange(0, len(ann_df_sorted), max(500, len(ann_df_sorted) // 5))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f'{int(x)}' for x in tick_positions], fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'in_hop1_vs_not_distribution_smoothed.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved smoothed in_hop1 vs not_in_hop1 distribution plot")
    
    # 3. Summary statistics box plots
    plt.figure(figsize=(14, 8))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, ann_type in enumerate(annotation_types):
        ax = axes[idx]
        
        ann_df = df[df[f"{ann_type}_total_de"] > 0].copy()
        
        if len(ann_df) == 0:
            ax.text(0.5, 0.5, f'No data for {ann_type}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{ann_type.upper()}', fontsize=12, fontweight='bold')
            continue
        
        # Prepare data for box plot
        plot_data = []
        labels = []
        
        in_hop1_vals = ann_df[f"{ann_type}_in_hop1"].values
        not_in_hop1_vals = ann_df[f"{ann_type}_not_in_hop1"].values
        
        plot_data.extend([in_hop1_vals, not_in_hop1_vals])
        labels.extend(['In Hop 1', 'Not in Hop 1'])
        
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')
        
        ax.set_ylabel('Number of Genes', fontsize=10)
        ax.set_title(f'{ann_type.upper()}\n(n={len(ann_df)} perturbations)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gene_count_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved gene count boxplots")
    
    # 4. Ratio distribution (deconvoluted ratio)
    plt.figure(figsize=(12, 6))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, ann_type in enumerate(annotation_types):
        ax = axes[idx]
        
        ann_df = df[df[f"{ann_type}_total_de"] > 0].copy()
        
        if len(ann_df) == 0:
            ax.text(0.5, 0.5, f'No data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'{ann_type.upper()}', fontsize=11, fontweight='bold')
            continue
        
        ratios = ann_df[f"{ann_type}_ratio"].values
        ax.hist(ratios, bins=30, color=sns.color_palette("husl", 1)[0], alpha=0.7, edgecolor='black')
        ax.axvline(ratios.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ratios.mean():.3f}')
        ax.axvline(np.median(ratios), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(ratios):.3f}')
        
        ax.set_xlabel('Deconvoluted Ratio', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'{ann_type.upper()}\n(n={len(ann_df)})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'deconvoluted_ratio_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved deconvoluted ratio distribution plot")


def print_summary_statistics(df: pd.DataFrame):
    """
    Print summary statistics to console.
    
    Args:
        df: Analysis DataFrame
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal perturbations analyzed: {len(df)}")
    
    print("\nBest Annotation Type Distribution:")
    print("-" * 80)
    best_ann_counts = df['best_annotation'].value_counts()
    for ann_type, count in best_ann_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {ann_type.upper():<12} {count:>6} ({pct:>5.1f}%)")
    
    print("\nOverall Statistics (Best Annotation):")
    print("-" * 80)
    print(f"  Mean total DE genes: {df['best_total_de'].mean():.2f}")
    print(f"  Mean in hop 1: {df['best_in_hop1'].mean():.2f}")
    print(f"  Mean not in hop 1: {df['best_not_in_hop1'].mean():.2f}")
    print(f"  Mean deconvoluted ratio: {df['best_deconvoluted_ratio'].mean():.3f}")
    
    print("\nStatistics by Annotation Type:")
    print("-" * 80)
    annotation_types = ["reactome", "bp", "cc", "mf"]
    
    for ann_type in annotation_types:
        ann_df = df[df[f"{ann_type}_total_de"] > 0]
        if len(ann_df) == 0:
            continue
        
        print(f"\n{ann_type.upper()}:")
        print(f"  Perturbations with data: {len(ann_df)}")
        print(f"  Mean total DE: {ann_df[f'{ann_type}_total_de'].mean():.2f}")
        print(f"  Mean in hop 1: {ann_df[f'{ann_type}_in_hop1'].mean():.2f}")
        print(f"  Mean not in hop 1: {ann_df[f'{ann_type}_not_in_hop1'].mean():.2f}")
        print(f"  Mean deconvoluted ratio: {ann_df[f'{ann_type}_ratio'].mean():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze deconvoluted genes output and create visualizations"
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        required=True,
        help="Path to deconvoluted_genes JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save output plots and CSV",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Path to save analysis CSV (default: output_dir/analysis_results.csv)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Load data
    data = load_deconvoluted_data(args.input_json)
    
    # Analyze data
    logging.info("Analyzing deconvoluted data...")
    df = analyze_deconvoluted_data(data)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Save CSV
    output_csv = args.output_csv if args.output_csv else args.output_dir / "analysis_results.csv"
    logging.info(f"\nSaving analysis results to {output_csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    # Create plots
    logging.info("\nCreating visualization plots...")
    create_distribution_plots(df, args.output_dir)
    
    logging.info("\nDone!")


if __name__ == "__main__":
    main()

