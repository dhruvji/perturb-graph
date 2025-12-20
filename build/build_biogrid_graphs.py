#!/usr/bin/env python3
"""
Build hop graphs for GBM perturbations using BioGRID interactions.

Interactions are loaded from the formatted MITAB file, filtered to genes that
are knocked out in the GBM datasets (targets in the merged CSV). For each
target gene, we record which other knockout genes are reachable within a given
number of hops. Only genes present in the dataset's feature column are kept in
the hop lists so the output matches the GO-based graph format.
"""

import argparse
import json
import logging
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


def load_csv_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV file and ensure required columns exist."""
    logging.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"target", "feature"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV file missing required columns: {sorted(missing)}")
    return df


def extract_targets_and_features(df: pd.DataFrame) -> (Set[str], Set[str]):
    """Return uppercase target genes (knockouts) and feature genes."""
    targets = (
        df["target"].astype(str).str.upper().str.strip().replace({"": None}).dropna().unique()
    )
    features = (
        df["feature"].astype(str).str.upper().str.strip().replace({"": None}).dropna().unique()
    )
    return set(targets), set(features)


def parse_biogrid_interactions(
    biogrid_path: Path, allowed_genes: Set[str]
) -> Dict[str, Set[str]]:
    """
    Parse BioGRID formatted MITAB file and build an undirected adjacency list.

    Only interactions where both genes are in allowed_genes are retained so the
    graph focuses on GBM knockout genes.
    """
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    allowed_upper = {g.upper() for g in allowed_genes}

    # Example line:
    # KNOWN INTERACTION: MAP2K4 (...) interacts with FLNC (...)
    line_pattern = re.compile(
        r"KNOWN INTERACTION:\s*([^\s(]+)\s*\([^)]*\)\s*interacts with\s*([^\s(]+)\s*\(",
        re.IGNORECASE,
    )

    logging.info(f"Reading BioGRID interactions from {biogrid_path}")
    with open(biogrid_path, "r", encoding="utf-8") as f:
        for line in f:
            match = line_pattern.search(line)
            if not match:
                continue
            gene_a = match.group(1).upper()
            gene_b = match.group(2).upper()

            # Keep only interactions fully within the knockout set
            if gene_a not in allowed_upper or gene_b not in allowed_upper:
                continue

            adjacency[gene_a].add(gene_b)
            adjacency[gene_b].add(gene_a)

    # Ensure all allowed genes are present, even if isolated
    for gene in allowed_upper:
        adjacency.setdefault(gene, set())

    logging.info(f"Built BioGRID adjacency for {len(adjacency)} genes")
    return adjacency


def neighbors_by_hop(graph: Dict[str, Set[str]], start: str, max_hops: int) -> Dict[int, Set[str]]:
    """Breadth-first search to collect neighbors by hop distance."""
    start = start.upper()
    visited = {start}
    queue = deque([(start, 0)])
    hops: Dict[int, Set[str]] = {i: set() for i in range(1, max_hops + 1)}

    while queue:
        gene, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbor in graph.get(gene, set()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            hop_idx = depth + 1
            hops[hop_idx].add(neighbor)
            queue.append((neighbor, hop_idx))
    return hops


def build_biogrid_graphs(
    csv_path: Path,
    biogrid_path: Path,
    output_path: Path,
    max_hops: int = 10,
) -> None:
    """Build hop graphs for all targets using BioGRID interactions."""
    df = load_csv_data(csv_path)
    targets, features = extract_targets_and_features(df)
    logging.info(f"Found {len(targets)} knockout targets and {len(features)} feature genes")

    adjacency = parse_biogrid_interactions(biogrid_path, targets)

    results: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for idx, target in enumerate(sorted(targets), 1):
        hop_sets = neighbors_by_hop(adjacency, target, max_hops)
        # Keep only genes present in the dataset features to mirror GO output
        hop_lists = {str(h): sorted(list(genes & features)) for h, genes in hop_sets.items()}
        results[target.upper()] = {"biogrid": hop_lists}

        if idx % 25 == 0 or idx == len(targets):
            logging.info(f"Processed {idx}/{len(targets)} targets")

    logging.info(f"Saving BioGRID hop graphs to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Build BioGRID-based hop graphs for GBM perturbations"
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Merged GBM CSV with 'target' and 'feature' columns",
    )
    parser.add_argument(
        "--biogrid_path",
        type=Path,
        default=Path("/data/dhruvgautam/agent/sources/biogrid/BIOGRID-ALL-5.0.250-FORMATTED.mitab.txt"),
        help="Path to formatted BioGRID MITAB file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save BioGRID hop graph JSON",
    )
    parser.add_argument(
        "--max_hops",
        type=int,
        default=10,
        help="Maximum hop distance to explore (default: 10)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    build_biogrid_graphs(
        csv_path=args.input_csv,
        biogrid_path=args.biogrid_path,
        output_path=args.output,
        max_hops=args.max_hops,
    )


if __name__ == "__main__":
    main()
