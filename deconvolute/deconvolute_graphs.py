#!/usr/bin/env python3
"""
Deconvolute differentially expressed genes by removing direct pathway connections.

For each perturbation, finds genes that are differentially expressed (FDR < 0.05)
but are NOT in the first hop for any of the 4 annotation categories (reactome, bp, cc, mf).
These represent indirect effects that are not directly connected via pathways.
"""

import argparse
import json
import logging
import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import pandas as pd

# Global variables for worker processes
_worker_graphs_data = None
_worker_de_genes_by_target = None
_worker_fold_changes = None


def load_csv_data(
    csv_path: Path, 
    fdr_threshold: float = 0.05,
    min_reference_mean: float = None,
    min_target_mean: float = None,
    max_fold_change: float = None
) -> pd.DataFrame:
    """
    Load CSV file and filter for differentially expressed genes.
    
    Args:
        csv_path: Path to CSV file
        fdr_threshold: FDR threshold for differential expression (default: 0.05)
        min_reference_mean: Minimum reference/control mean expression (filters inflated fold changes)
        min_target_mean: Minimum target/perturbation mean expression
        max_fold_change: Maximum fold change to consider (caps inflated values from low control expression)
        
    Returns:
        DataFrame filtered for DE genes
    """
    logging.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check required columns exist
    required_cols = ['target', 'feature', 'fdr']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file missing required columns: {missing_cols}")
    
    # Filter for DE genes (FDR < threshold)
    df_de = df[df['fdr'] < fdr_threshold].copy()
    initial_count = len(df_de)
    logging.info(f"Found {initial_count} DE gene entries (FDR < {fdr_threshold})")
    
    # Apply expression thresholds to filter inflated fold changes
    if min_reference_mean is not None:
        if 'reference_mean' not in df_de.columns:
            logging.warning("'reference_mean' column not found, skipping min_reference_mean filter")
        else:
            before = len(df_de)
            df_de = df_de[df_de['reference_mean'] >= min_reference_mean].copy()
            filtered = before - len(df_de)
            logging.info(f"Filtered {filtered} genes with reference_mean < {min_reference_mean} ({len(df_de)} remaining)")
    
    if min_target_mean is not None:
        if 'target_mean' not in df_de.columns:
            logging.warning("'target_mean' column not found, skipping min_target_mean filter")
        else:
            before = len(df_de)
            df_de = df_de[df_de['target_mean'] >= min_target_mean].copy()
            filtered = before - len(df_de)
            logging.info(f"Filtered {filtered} genes with target_mean < {min_target_mean} ({len(df_de)} remaining)")
    
    if max_fold_change is not None:
        if 'fold_change' not in df_de.columns:
            logging.warning("'fold_change' column not found, skipping max_fold_change filter")
        else:
            before = len(df_de)
            # Use absolute fold change for filtering
            abs_fold_change = df_de['fold_change'].abs()
            df_de = df_de[abs_fold_change <= max_fold_change].copy()
            filtered = before - len(df_de)
            logging.info(f"Filtered {filtered} genes with |fold_change| > {max_fold_change} ({len(df_de)} remaining)")
    
    logging.info(f"Final filtered count: {len(df_de)} DE gene entries (from {initial_count})")
    
    return df_de


def extract_de_genes_by_target(df_de: pd.DataFrame, top_n: int = None) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], float]]:
    """
    Extract DE genes grouped by target gene, optionally filtering to top N per target.
    
    Args:
        df_de: DataFrame filtered for DE genes
        top_n: If provided, only keep top N genes per target based on rank (lower rank = higher priority)
        
    Returns:
        Tuple of:
        - Dictionary mapping target gene (uppercase) -> set of DE feature genes (uppercase)
        - Dictionary mapping (target_upper, feature_upper) -> fold_change
    """
    de_genes_by_target: Dict[str, Set[str]] = defaultdict(set)
    fold_changes: Dict[Tuple[str, str], float] = {}
    
    if top_n is not None and top_n > 0:
        # Sort by rank (lower rank = higher priority) and take top N per target
        if 'rank' not in df_de.columns:
            logging.warning("'rank' column not found, using abs_fold_change for ranking")
            if 'abs_fold_change' in df_de.columns:
                df_sorted = df_de.sort_values('abs_fold_change', ascending=False)
            else:
                logging.warning("No ranking column found, using all DE genes")
                df_sorted = df_de
        else:
            df_sorted = df_de.sort_values('rank', ascending=True)
        
        # Group by target and take top N
        for target, group in df_sorted.groupby('target', sort=False):
            target_upper = str(target).strip().upper()
            top_genes = group.head(top_n)
            
            for _, row in top_genes.iterrows():
                feature = str(row['feature']).strip().upper()
                if target_upper and feature:
                    de_genes_by_target[target_upper].add(feature)
                    # Store fold change if available
                    if 'fold_change' in row and pd.notna(row['fold_change']):
                        fold_changes[(target_upper, feature)] = float(row['fold_change'])
        
        logging.info(f"Found {len(de_genes_by_target)} target genes with DE genes (top {top_n} per target)")
    else:
        # Use all DE genes
        for _, row in df_de.iterrows():
            target = str(row['target']).strip().upper()
            feature = str(row['feature']).strip().upper()
            
            if target and feature:
                de_genes_by_target[target].add(feature)
                # Store fold change if available
                if 'fold_change' in row and pd.notna(row['fold_change']):
                    fold_changes[(target, feature)] = float(row['fold_change'])
        
        logging.info(f"Found {len(de_genes_by_target)} target genes with DE genes")
    
    return dict(de_genes_by_target), fold_changes


def load_graphs_data(json_path: Path) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    Load GO graphs JSON file.
    
    Args:
        json_path: Path to go_graphs.json file
        
    Returns:
        Dictionary with graph data structure
    """
    logging.info(f"Loading graphs from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    logging.info(f"Loaded graphs for {len(data)} target genes")
    return data


def _init_worker(
    graphs_data: Dict[str, Dict[str, Dict[str, List[str]]]],
    de_genes_by_target: Dict[str, Set[str]],
    fold_changes: Dict[Tuple[str, str], float]
) -> None:
    """Initialize worker process with graphs data, DE genes, and fold changes."""
    global _worker_graphs_data, _worker_de_genes_by_target, _worker_fold_changes
    _worker_graphs_data = graphs_data
    _worker_de_genes_by_target = de_genes_by_target
    _worker_fold_changes = fold_changes


def process_single_target(target_gene: str) -> Tuple[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """
    Process a single target gene to find deconvoluted DE genes.
    
    A gene is "deconvoluted" if it is:
    - Differentially expressed (FDR < 0.05)
    - NOT in hop 1 for any of the 4 annotation categories
    
    Args:
        target_gene: Target gene symbol
        
    Returns:
        Tuple of (target_gene_upper, deconvoluted_genes_dict)
        Each gene in the output is represented as {"gene": "GENE1", "fold_change": 2.5}
    """
    global _worker_graphs_data, _worker_de_genes_by_target, _worker_fold_changes
    
    target_gene_upper = target_gene.upper()
    logging.info(f"Processing target: {target_gene_upper}")
    
    # Get DE genes for this target
    de_genes = _worker_de_genes_by_target.get(target_gene_upper, set())
    
    if not de_genes:
        logging.info(f"  No DE genes found for {target_gene_upper}")
        return target_gene_upper, {}
    
    # Get hop 1 genes from all annotation types
    graphs_for_target = _worker_graphs_data.get(target_gene_upper, {})
    
    hop1_genes_all_types: Set[str] = set()
    annotation_types = ["reactome", "bp", "cc", "mf"]
    
    for ann_type in annotation_types:
        if ann_type in graphs_for_target:
            hop1_list = graphs_for_target[ann_type].get("1", [])
            hop1_genes_all_types.update(gene.upper() for gene in hop1_list)
    
    logging.info(f"  Found {len(de_genes)} DE genes, {len(hop1_genes_all_types)} genes in hop 1 across all types")
    
    # Find DE genes that are NOT in hop 1 for any annotation type
    deconvoluted_genes = de_genes - hop1_genes_all_types
    
    logging.info(f"  Found {len(deconvoluted_genes)} deconvoluted genes (not in hop 1)")
    
    # Helper function to create gene entry with fold change
    def make_gene_entry(gene: str) -> Dict[str, Any]:
        entry = {"gene": gene}
        fold_change = _worker_fold_changes.get((target_gene_upper, gene))
        if fold_change is not None:
            entry["fold_change"] = fold_change
        return entry
    
    # Organize by annotation type for output
    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    
    # Store deconvoluted genes (not in hop 1 for ANY annotation type)
    # Sort by fold_change (descending), then by gene name (ascending)
    result["deconvoluted"] = sorted(
        [make_gene_entry(gene) for gene in deconvoluted_genes],
        key=lambda x: (x.get("fold_change") if x.get("fold_change") is not None else float('-inf'), x["gene"]),
        reverse=True
    )
    
    # For each annotation type, show which DE genes are in/not in hop 1
    for ann_type in annotation_types:
        if ann_type in graphs_for_target:
            hop1_genes = set(
                gene.upper() 
                for gene in graphs_for_target[ann_type].get("1", [])
            )
            # DE genes that are in hop 1 for this specific annotation type
            in_hop1 = sorted(
                [make_gene_entry(gene) for gene in (de_genes & hop1_genes)],
                key=lambda x: (x.get("fold_change") if x.get("fold_change") is not None else float('-inf'), x["gene"]),
                reverse=True
            )
            # DE genes that are NOT in hop 1 for this annotation type
            not_in_hop1 = sorted(
                [make_gene_entry(gene) for gene in (de_genes - hop1_genes)],
                key=lambda x: (x.get("fold_change") if x.get("fold_change") is not None else float('-inf'), x["gene"]),
                reverse=True
            )
            
            result[ann_type] = {
                "in_hop1": in_hop1,
                "not_in_hop1": not_in_hop1
            }
        else:
            # If no graph data, all DE genes are not in hop 1
            result[ann_type] = {
                "in_hop1": [],
                "not_in_hop1": sorted(
                    [make_gene_entry(gene) for gene in de_genes],
                    key=lambda x: (x.get("fold_change") if x.get("fold_change") is not None else float('-inf'), x["gene"]),
                    reverse=True
                )
            }
    
    return target_gene_upper, result


def deconvolute_graphs(
    csv_path: Path,
    graphs_json_path: Path,
    output_path: Path,
    fdr_threshold: float = 0.05,
    top_n: int = None,
    num_workers: int = 1,
    min_reference_mean: float = None,
    min_target_mean: float = None,
    max_fold_change: float = None
) -> None:
    """
    Deconvolute differentially expressed genes by removing direct pathway connections.
    
    Args:
        csv_path: Path to CSV file with DE results
        graphs_json_path: Path to go_graphs.json file
        output_path: Path to save output JSON file
        fdr_threshold: FDR threshold for differential expression
        top_n: If provided, only use top N DE genes per target (based on rank)
        num_workers: Number of parallel workers to use
        min_reference_mean: Minimum reference/control mean expression (filters inflated fold changes)
        min_target_mean: Minimum target/perturbation mean expression
        max_fold_change: Maximum fold change to consider (caps inflated values from low control expression)
    """
    # Load CSV and filter for DE genes
    df_de = load_csv_data(
        csv_path, 
        fdr_threshold,
        min_reference_mean=min_reference_mean,
        min_target_mean=min_target_mean,
        max_fold_change=max_fold_change
    )
    
    if top_n is not None and top_n > 0:
        logging.info(f"Filtering to top {top_n} DE genes per target based on rank")
    
    # Extract DE genes by target (optionally filtered to top N) and fold changes
    de_genes_by_target, fold_changes = extract_de_genes_by_target(df_de, top_n=top_n)
    
    # Load graphs data
    graphs_data = load_graphs_data(graphs_json_path)
    
    # Get all target genes (union of CSV targets and graph targets)
    csv_targets = set(de_genes_by_target.keys())
    graph_targets = set(graphs_data.keys())
    all_targets = sorted(csv_targets | graph_targets)
    
    logging.info(f"\nProcessing {len(all_targets)} target genes using {num_workers} workers...")
    logging.info(f"Loaded fold change data for {len(fold_changes)} gene-target pairs")
    
    results: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    
    if num_workers == 1:
        # Sequential processing
        # Initialize worker data for sequential mode
        _init_worker(graphs_data, de_genes_by_target, fold_changes)
        for idx, target_gene in enumerate(all_targets, 1):
            target_gene_upper, result = process_single_target(target_gene)
            results[target_gene_upper] = result
            
            if idx % 100 == 0:
                logging.info(f"  Processed {idx}/{len(all_targets)} target genes...")
    else:
        # Parallel processing using multiprocessing.Pool
        completed = 0
        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(graphs_data, de_genes_by_target, fold_changes)
        ) as pool:
            # Process all target genes in parallel
            try:
                for target_gene_upper, result in pool.imap_unordered(
                    process_single_target, all_targets
                ):
                    results[target_gene_upper] = result
                    completed += 1
                    
                    if completed % 100 == 0:
                        logging.info(f"  Processed {completed}/{len(all_targets)} target genes...")
            except Exception as e:
                logging.error(f"Error during parallel processing: {e}")
                raise
    
    logging.info(f"\nProcessed {len(results)} target genes")
    
    # Calculate summary statistics
    total_deconvoluted = sum(
        len(result.get("deconvoluted", []))
        for result in results.values()
    )
    targets_with_deconvoluted = sum(
        1 for result in results.values()
        if len(result.get("deconvoluted", [])) > 0
    )
    
    logging.info(f"\nSummary:")
    logging.info(f"  Total deconvoluted genes: {total_deconvoluted}")
    logging.info(f"  Targets with deconvoluted genes: {targets_with_deconvoluted}")
    logging.info(f"  Note: Gene entries now include fold_change values when available")
    
    # Save results
    logging.info(f"\nSaving results to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Deconvolute differentially expressed genes by removing direct pathway connections"
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Path to input CSV file with DE results (e.g., /data/dhruvgautam/chemogenetic_h1/DMSO/ranked/complete_ranked_DMSO.csv)",
    )
    parser.add_argument(
        "--graphs_json",
        type=Path,
        required=True,
        help="Path to go_graphs.json file (e.g., /data/dhruvgautam/chemogenetic_h1/DMSO/h5/go_graphs.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save output JSON file",
    )
    parser.add_argument(
        "--fdr_threshold",
        type=float,
        default=0.05,
        help="FDR threshold for differential expression (default: 0.05)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="If provided, only use top N DE genes per target based on rank (default: None, use all DE genes)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers to use (default: 1)",
    )
    parser.add_argument(
        "--min_reference_mean",
        type=float,
        default=None,
        help="Minimum reference/control mean expression to filter inflated fold changes (e.g., 0.5 or 1.0 for shallow sequencing)",
    )
    parser.add_argument(
        "--min_target_mean",
        type=float,
        default=None,
        help="Minimum target/perturbation mean expression (e.g., 0.5 or 1.0)",
    )
    parser.add_argument(
        "--max_fold_change",
        type=float,
        default=None,
        help="Maximum fold change to consider (caps inflated values from low control expression, e.g., 10.0)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    deconvolute_graphs(
        csv_path=args.input_csv,
        graphs_json_path=args.graphs_json,
        output_path=args.output,
        fdr_threshold=args.fdr_threshold,
        top_n=args.top_n,
        num_workers=args.num_workers,
        min_reference_mean=args.min_reference_mean,
        min_target_mean=args.min_target_mean,
        max_fold_change=args.max_fold_change
    )


if __name__ == "__main__":
    main()

