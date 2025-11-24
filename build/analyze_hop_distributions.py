#!/usr/bin/env python3
"""
Analyze the distribution of number of genes in each hop container for each gene ontology list.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def analyze_hop_distributions(json_path: Path) -> pd.DataFrame:
    """
    Analyze the distribution of gene counts per hop level for each annotation type.
    
    Args:
        json_path: Path to the go_graphs.json file
        
    Returns:
        DataFrame with statistics for each annotation type and hop level
    """
    print(f"Loading JSON from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotation_types = ["reactome", "bp", "cc", "mf"]
    
    # Collect gene counts per hop level for each annotation type
    # Structure: annotation_type -> hop_level -> list of gene counts
    counts_by_type_and_hop: Dict[str, Dict[int, List[int]]] = {
        ann_type: defaultdict(list) for ann_type in annotation_types
    }
    
    # Iterate through all target genes
    for target_gene, results in data.items():
        for ann_type in annotation_types:
            if ann_type not in results:
                continue
            
            for hop_str, genes in results[ann_type].items():
                hop = int(hop_str)
                gene_count = len(genes)
                counts_by_type_and_hop[ann_type][hop].append(gene_count)
    
    # Calculate statistics for each annotation type and hop level
    stats_rows = []
    
    for ann_type in annotation_types:
        for hop in sorted(counts_by_type_and_hop[ann_type].keys()):
            counts = counts_by_type_and_hop[ann_type][hop]
            
            if len(counts) == 0:
                continue
            
            stats_rows.append({
                'annotation_type': ann_type,
                'hop': hop,
                'n_targets': len(counts),
                'mean': np.mean(counts),
                'median': np.median(counts),
                'std': np.std(counts),
                'min': np.min(counts),
                'max': np.max(counts),
                'p25': np.percentile(counts, 25),
                'p75': np.percentile(counts, 75),
                'p90': np.percentile(counts, 90),
                'p95': np.percentile(counts, 95),
                'p99': np.percentile(counts, 99),
            })
    
    df = pd.DataFrame(stats_rows)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze distribution of genes per hop level for each annotation type"
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        required=True,
        help="Path to input go_graphs.json file",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Path to save output CSV with statistics",
    )
    
    args = parser.parse_args()
    
    # Analyze distributions
    df = analyze_hop_distributions(args.input_json)
    
    # Save results
    print(f"\nSaving statistics to {args.output_csv}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for ann_type in ["reactome", "bp", "cc", "mf"]:
        ann_df = df[df['annotation_type'] == ann_type]
        if len(ann_df) == 0:
            continue
        
        print(f"\n{ann_type.upper()}:")
        print("-" * 80)
        print(f"{'Hop':<6} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<8} {'Max':<10} {'P95':<10} {'N Targets':<12}")
        print("-" * 80)
        
        for _, row in ann_df.iterrows():
            print(f"{int(row['hop']):<6} "
                  f"{row['mean']:<12.2f} "
                  f"{row['median']:<12.2f} "
                  f"{row['std']:<12.2f} "
                  f"{int(row['min']):<8} "
                  f"{int(row['max']):<10} "
                  f"{row['p95']:<10.2f} "
                  f"{int(row['n_targets']):<12}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

