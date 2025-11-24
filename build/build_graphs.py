#!/usr/bin/env python3
"""
Build graphs from gene ontology annotations for target gene perturbations.

For each annotation type (reactome, bp, cc, mf), builds a graph where genes
are connected if they share pathways/categories. Then for each target gene
perturbation, finds genes in the feature column that are at 
1-hop, 2-hop, 3-hop, etc. distances in the GO/Reactome pathway graphs.
"""

import argparse
import json
import logging
import multiprocessing
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# Global variable for worker processes to access graphs
_worker_graphs = None
_worker_max_hops = None
_worker_feature_genes = None


def load_annotation_sets(go_dir: Path) -> Dict[str, Dict[str, Set[str]]]:
    """
    Load GO and Reactome gene sets from MSigDB JSON exports.

    Returns a nested mapping:
      collection_key -> gene_symbol_upper -> set of pathway names

    Where collection_key is one of: "reactome", "bp", "cc", "mf".
    """
    collection_files = {
        "reactome": go_dir / "c2.cp.reactome.v2025.1.Hs.json",
        "bp": go_dir / "c5.go.bp.v2025.1.Hs.json",
        "cc": go_dir / "c5.go.cc.v2025.1.Hs.json",
        "mf": go_dir / "c5.go.mf.v2025.1.Hs.json",
    }

    collection_to_gene_to_paths: Dict[str, Dict[str, Set[str]]] = {
        key: {} for key in collection_files.keys()
    }

    for collection_key, json_path in collection_files.items():
        if not json_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {json_path}")

        # File structure is a single JSON object: { pathway_name: { geneSymbols: [...] , ... }, ... }
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        gene_to_paths: Dict[str, Set[str]] = collection_to_gene_to_paths[collection_key]

        for pathway_name, meta in data.items():
            genes: List[str] = meta.get("geneSymbols", [])
            for gene_symbol in genes:
                gene_upper = str(gene_symbol).strip().upper()
                if not gene_upper:
                    continue
                if gene_upper not in gene_to_paths:
                    gene_to_paths[gene_upper] = set()
                gene_to_paths[gene_upper].add(pathway_name)

    return collection_to_gene_to_paths


def build_gene_graph(gene_to_paths: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """
    Build a graph where genes are connected if they share at least one pathway.

    Args:
        gene_to_paths: Mapping from gene (uppercase) to set of pathway names

    Returns:
        Dictionary mapping gene -> set of connected genes
    """
    # Build pathway -> genes mapping
    pathway_to_genes: Dict[str, Set[str]] = defaultdict(set)
    for gene, pathways in gene_to_paths.items():
        for pathway in pathways:
            pathway_to_genes[pathway].add(gene)

    # Build gene graph: two genes are connected if they share a pathway
    graph: Dict[str, Set[str]] = defaultdict(set)
    for pathway, genes_in_pathway in pathway_to_genes.items():
        genes_list = list(genes_in_pathway)
        # Connect all pairs of genes in the same pathway
        for i, gene1 in enumerate(genes_list):
            for gene2 in genes_list[i + 1:]:
                graph[gene1].add(gene2)
                graph[gene2].add(gene1)

    return dict(graph)


def find_neighbors_at_hops(
    graph: Dict[str, Set[str]], 
    start_gene: str, 
    max_hops: int = 3
) -> Dict[int, Set[str]]:
    """
    Find all neighbors at different hop distances from a starting gene.

    Args:
        graph: Gene graph (gene -> set of connected genes)
        start_gene: Starting gene (uppercase)
        max_hops: Maximum number of hops to explore

    Returns:
        Dictionary mapping hop distance -> set of genes at that distance
    """
    start_gene_upper = start_gene.upper()
    
    if start_gene_upper not in graph:
        return {i: set() for i in range(1, max_hops + 1)}

    # BFS to find neighbors at each hop distance
    visited = {start_gene_upper}
    neighbors_by_hop: Dict[int, Set[str]] = {i: set() for i in range(1, max_hops + 1)}
    
    queue = deque([(start_gene_upper, 0)])
    
    while queue:
        current_gene, current_hop = queue.popleft()
        
        if current_hop >= max_hops:
            continue
        
        # Explore neighbors
        for neighbor in graph.get(current_gene, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                next_hop = current_hop + 1
                neighbors_by_hop[next_hop].add(neighbor)
                queue.append((neighbor, next_hop))
    
    return neighbors_by_hop


def load_csv_data(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV file with target and feature columns.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with target and feature columns
    """
    logging.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check required columns exist
    required_cols = ['target', 'feature']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file missing required columns: {missing_cols}")
    
    return df


def extract_target_genes(df: pd.DataFrame) -> Set[str]:
    """
    Extract unique target gene perturbations from CSV.

    Args:
        df: DataFrame with 'target' column

    Returns:
        Set of target gene symbols (uppercase)
    """
    target_genes = set()
    genes = df['target'].astype(str).str.upper().str.strip()
    target_genes.update(genes.dropna().unique())
    
    # Remove empty strings
    target_genes.discard('')
    
    return target_genes


def extract_feature_genes(df: pd.DataFrame) -> Set[str]:
    """
    Extract unique feature genes from CSV (the genes measured in the dataset).

    Args:
        df: DataFrame with 'feature' column

    Returns:
        Set of feature gene symbols (uppercase)
    """
    feature_genes = set()
    genes = df['feature'].astype(str).str.upper().str.strip()
    feature_genes.update(genes.dropna().unique())
    
    # Remove empty strings
    feature_genes.discard('')
    
    return feature_genes


def _init_worker(
    graphs_by_type: Dict[str, Dict[str, Set[str]]], 
    max_hops: int,
    feature_genes: Set[str]
) -> None:
    """Initialize worker process with graphs, max_hops, and feature genes."""
    global _worker_graphs, _worker_max_hops, _worker_feature_genes
    _worker_graphs = graphs_by_type
    _worker_max_hops = max_hops
    _worker_feature_genes = feature_genes


def process_single_target_gene(target_gene: str) -> Tuple[str, Dict[str, Dict[int, List[str]]]]:
    """
    Process a single target gene perturbation.
    Uses global _worker_graphs, _worker_max_hops, and _worker_feature_genes set by _init_worker.
    Filters results to only include genes that are in the feature column.
    
    Args:
        target_gene: Target gene symbol
    
    Returns:
        Tuple of (target_gene_upper, results_dict)
    """
    global _worker_graphs, _worker_max_hops, _worker_feature_genes
    
    target_gene_upper = target_gene.upper()
    logging.info(f"Processing perturbation: {target_gene_upper}")
    
    result: Dict[str, Dict[int, List[str]]] = {}
    
    for ann_type in ["reactome", "bp", "cc", "mf"]:
        graph = _worker_graphs[ann_type]
        neighbors_by_hop = find_neighbors_at_hops(graph, target_gene_upper, _worker_max_hops)
        
        # Filter neighbors to only include genes in the feature column
        # Convert sets to sorted lists for JSON serialization
        result[ann_type] = {
            hop: sorted(neighbors & _worker_feature_genes) 
            for hop, neighbors in neighbors_by_hop.items()
        }
    
    logging.info(f"Completed perturbation: {target_gene_upper}")
    return target_gene_upper, result


def build_graphs_for_perturbations(
    csv_path: Path,
    go_dir: Path,
    output_path: Path,
    max_hops: int = 3,
    num_workers: int = 1
) -> None:
    """
    Build graphs for all target gene perturbations using GO/Reactome annotations.
    
    For each target gene perturbation, finds genes in the feature column
    that are 1-hop, 2-hop, 3-hop, etc. away in the GO/Reactome pathway graphs.

    Args:
        csv_path: Path to input CSV file with 'target' and 'feature' columns
        go_dir: Directory containing GO/Reactome JSON files
        output_path: Path to save output JSON file
        max_hops: Maximum number of hops to explore
        num_workers: Number of parallel workers to use
    """
    # Load CSV data
    df = load_csv_data(csv_path)
    
    logging.info("Extracting target genes...")
    target_genes = extract_target_genes(df)
    logging.info(f"Found {len(target_genes)} unique target genes")
    
    logging.info("Extracting feature genes...")
    feature_genes = extract_feature_genes(df)
    logging.info(f"Found {len(feature_genes)} feature genes in dataset")
    
    logging.info("Loading annotation sets...")
    annotation_sets = load_annotation_sets(go_dir)
    
    # Build graphs for each annotation type
    graphs_by_type: Dict[str, Dict[str, Set[str]]] = {}
    for ann_type, gene_to_paths in annotation_sets.items():
        logging.info(f"Building graph for {ann_type}...")
        graphs_by_type[ann_type] = build_gene_graph(gene_to_paths)
        logging.info(f"  Graph has {len(graphs_by_type[ann_type])} genes with connections")
    
    # Build graphs for each target gene
    logging.info(f"\nBuilding graphs for target genes using {num_workers} workers...")
    results: Dict[str, Dict[str, Dict[int, List[str]]]] = {}
    
    sorted_target_genes = sorted(target_genes)
    
    if num_workers == 1:
        # Sequential processing
        for idx, target_gene in enumerate(sorted_target_genes, 1):
            # Set globals for sequential processing
            global _worker_graphs, _worker_max_hops, _worker_feature_genes
            _worker_graphs = graphs_by_type
            _worker_max_hops = max_hops
            _worker_feature_genes = feature_genes
            
            target_gene_upper, result = process_single_target_gene(target_gene)
            results[target_gene_upper] = result
            
            if idx % 100 == 0:
                logging.info(f"  Processed {idx}/{len(sorted_target_genes)} target genes...")
    else:
        # Parallel processing using multiprocessing.Pool
        completed = 0
        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(graphs_by_type, max_hops, feature_genes)
        ) as pool:
            # Process all target genes in parallel
            try:
                for target_gene_upper, result in pool.imap_unordered(
                    process_single_target_gene, sorted_target_genes
                ):
                    results[target_gene_upper] = result
                    completed += 1
                    
                    if completed % 100 == 0:
                        logging.info(f"  Processed {completed}/{len(sorted_target_genes)} target genes...")
            except Exception as e:
                logging.error(f"Error during parallel processing: {e}")
                raise
    
    logging.info(f"\nProcessed {len(results)} target genes")
    
    # Save results
    logging.info(f"Saving results to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Build graphs from gene ontology annotations for target gene perturbations"
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Path to input CSV file with 'target' and 'feature' columns (e.g., /data/dhruvgautam/chemogenetic_h1/DMSO/ranked/complete_ranked_DMSO.csv)",
    )
    parser.add_argument(
        "--go_dir",
        type=Path,
        default=Path("/data/dhruvgautam/go"),
        help="Directory containing GO/Reactome JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save output JSON file",
    )
    parser.add_argument(
        "--max_hops",
        type=int,
        default=10,
        help="Maximum number of hops to explore (default: 10)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers to use (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    build_graphs_for_perturbations(
        csv_path=args.input_csv,
        go_dir=args.go_dir,
        output_path=args.output,
        max_hops=args.max_hops,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()

