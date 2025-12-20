#!/usr/bin/env python3
"""
Deconvolute GBM DE genes using BioGRID hop graphs.

For each perturbation, remove DE genes that fall within hop 1 of the BioGRID
graph built on the GBM knockout genes. Output mirrors the GO-based
deconvolution format but uses a single 'biogrid' annotation type.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd


def load_csv_data(
    csv_path: Path,
    fdr_threshold: float = 0.05,
    min_reference_mean: float = None,
    min_target_mean: float = None,
    max_fold_change: float = None,
) -> pd.DataFrame:
    """Load CSV and filter for differentially expressed genes with optional expression/FC thresholds."""
    logging.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"target", "feature", "fdr"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV file missing required columns: {sorted(missing)}")
    df_de = df[df["fdr"] < fdr_threshold].copy()
    logging.info(f"Found {len(df_de)} DE entries at FDR < {fdr_threshold}")

    if min_reference_mean is not None:
        if "reference_mean" in df_de.columns:
            before = len(df_de)
            df_de = df_de[df_de["reference_mean"] >= min_reference_mean].copy()
            logging.info(f"Filtered {before - len(df_de)} with reference_mean < {min_reference_mean}")
        else:
            logging.warning("reference_mean column missing; skipping min_reference_mean filter")

    if min_target_mean is not None:
        if "target_mean" in df_de.columns:
            before = len(df_de)
            df_de = df_de[df_de["target_mean"] >= min_target_mean].copy()
            logging.info(f"Filtered {before - len(df_de)} with target_mean < {min_target_mean}")
        else:
            logging.warning("target_mean column missing; skipping min_target_mean filter")

    if max_fold_change is not None:
        if "fold_change" in df_de.columns:
            before = len(df_de)
            df_de = df_de[df_de["fold_change"].abs() <= max_fold_change].copy()
            logging.info(f"Filtered {before - len(df_de)} with |fold_change| > {max_fold_change}")
        else:
            logging.warning("fold_change column missing; skipping max_fold_change filter")
    return df_de


def extract_de_genes_by_target(
    df_de: pd.DataFrame, top_n: int = None
) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], float]]:
    """
    Group DE genes by target; optionally keep top N per target based on rank.
    """
    de_genes_by_target: Dict[str, Set[str]] = defaultdict(set)
    fold_changes: Dict[Tuple[str, str], float] = {}

    if top_n is not None and top_n > 0:
        if "rank" in df_de.columns:
            df_sorted = df_de.sort_values("rank", ascending=True)
        elif "abs_fold_change" in df_de.columns:
            df_sorted = df_de.sort_values("abs_fold_change", ascending=False)
        else:
            df_sorted = df_de

        for target, group in df_sorted.groupby("target", sort=False):
            target_upper = str(target).strip().upper()
            for _, row in group.head(top_n).iterrows():
                feature = str(row["feature"]).strip().upper()
                if not feature:
                    continue
                de_genes_by_target[target_upper].add(feature)
                if "fold_change" in row and pd.notna(row["fold_change"]):
                    fold_changes[(target_upper, feature)] = float(row["fold_change"])
    else:
        for _, row in df_de.iterrows():
            target = str(row["target"]).strip().upper()
            feature = str(row["feature"]).strip().upper()
            if not target or not feature:
                continue
            de_genes_by_target[target].add(feature)
            if "fold_change" in row and pd.notna(row["fold_change"]):
                fold_changes[(target, feature)] = float(row["fold_change"])

    logging.info(f"Collected DE genes for {len(de_genes_by_target)} targets")
    return dict(de_genes_by_target), fold_changes


def load_graphs_data(json_path: Path) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Load BioGRID hop graphs from JSON."""
    logging.info(f"Loading BioGRID graphs from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info(f"Loaded graphs for {len(data)} targets")
    return data


def process_single_target(
    target_gene: str,
    graphs_data: Dict[str, Dict[str, Dict[str, List[str]]]],
    de_genes_by_target: Dict[str, Set[str]],
    fold_changes: Dict[Tuple[str, str], float],
    annotation_types: List[str],
) -> Tuple[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """Deconvolute a single target using provided graphs."""
    target_upper = target_gene.upper()
    de_genes = de_genes_by_target.get(target_upper, set())
    graphs_for_target = graphs_data.get(target_upper, {})

    hop1_genes_all: Set[str] = set()
    for ann_type in annotation_types:
        hop1_genes_all.update(g.upper() for g in graphs_for_target.get(ann_type, {}).get("1", []))

    deconvoluted_genes = de_genes - hop1_genes_all

    def make_entry(gene: str) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"gene": gene}
        fc = fold_changes.get((target_upper, gene))
        if fc is not None:
            entry["fold_change"] = fc
        return entry

    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    result["deconvoluted"] = sorted(
        [make_entry(g) for g in deconvoluted_genes],
        key=lambda x: (x.get("fold_change") if x.get("fold_change") is not None else float("-inf"), x["gene"]),
        reverse=True,
    )

    for ann_type in annotation_types:
        hop1 = set(g.upper() for g in graphs_for_target.get(ann_type, {}).get("1", []))
        in_hop1 = de_genes & hop1
        not_in_hop1 = de_genes - hop1
        result[ann_type] = {
            "in_hop1": sorted(
                [make_entry(g) for g in in_hop1],
                key=lambda x: (x.get("fold_change") if x.get("fold_change") is not None else float("-inf"), x["gene"]),
                reverse=True,
            ),
            "not_in_hop1": sorted(
                [make_entry(g) for g in not_in_hop1],
                key=lambda x: (x.get("fold_change") if x.get("fold_change") is not None else float("-inf"), x["gene"]),
                reverse=True,
            ),
        }

    return target_upper, result


def deconvolute_biogrid_graphs(
    csv_path: Path,
    graphs_json_path: Path,
    output_path: Path,
    fdr_threshold: float = 0.05,
    top_n: int = None,
    min_reference_mean: float = None,
    min_target_mean: float = None,
    max_fold_change: float = None,
) -> None:
    """Deconvolute DE genes using BioGRID hop graphs."""
    df_de = load_csv_data(
        csv_path,
        fdr_threshold=fdr_threshold,
        min_reference_mean=min_reference_mean,
        min_target_mean=min_target_mean,
        max_fold_change=max_fold_change,
    )
    de_genes_by_target, fold_changes = extract_de_genes_by_target(df_de, top_n=top_n)
    graphs_data = load_graphs_data(graphs_json_path)

    annotation_types = ["biogrid"]
    all_targets = sorted(set(de_genes_by_target.keys()) | set(graphs_data.keys()))
    logging.info(f"Processing {len(all_targets)} targets with annotation types: {annotation_types}")

    results: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    for idx, target in enumerate(all_targets, 1):
        target_upper, res = process_single_target(
            target, graphs_data, de_genes_by_target, fold_changes, annotation_types
        )
        results[target_upper] = res
        if idx % 50 == 0 or idx == len(all_targets):
            logging.info(f"Processed {idx}/{len(all_targets)} targets")

    total_deconvoluted = sum(len(res.get("deconvoluted", [])) for res in results.values())
    logging.info(f"Total deconvoluted genes: {total_deconvoluted}")

    logging.info(f"Saving deconvoluted results to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Deconvolute GBM DE genes using BioGRID hops")
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Merged GBM CSV with DE results",
    )
    parser.add_argument(
        "--graphs_json",
        type=Path,
        required=True,
        help="BioGRID hop graph JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save deconvoluted JSON",
    )
    parser.add_argument(
        "--fdr_threshold",
        type=float,
        default=0.05,
        help="FDR threshold for DE gene filtering",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="If set, keep only top N DE genes per target by rank (if available)",
    )
    parser.add_argument(
        "--min_reference_mean",
        type=float,
        default=None,
        help="Minimum reference/control mean expression to retain a gene",
    )
    parser.add_argument(
        "--min_target_mean",
        type=float,
        default=None,
        help="Minimum target/perturbation mean expression to retain a gene",
    )
    parser.add_argument(
        "--max_fold_change",
        type=float,
        default=None,
        help="Maximum absolute fold change to retain (filters inflated ratios)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    deconvolute_biogrid_graphs(
        csv_path=args.input_csv,
        graphs_json_path=args.graphs_json,
        output_path=args.output,
        fdr_threshold=args.fdr_threshold,
        top_n=args.top_n,
        min_reference_mean=args.min_reference_mean,
        min_target_mean=args.min_target_mean,
        max_fold_change=args.max_fold_change,
    )


if __name__ == "__main__":
    main()
