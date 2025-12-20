#!/usr/bin/env python3
"""Prepare ranked DE CSV for GSE250558 targeted perturb-seq dataset.

The raw combined results file includes multiple time points (6/12/18h) and
controls (SAFE, raf.transgene). This script:
  - Drops control guides
  - Collapses multiple entries per (guide, gene) by picking the row with the
    lowest adjusted p-value, breaking ties with highest |lfc|
  - Computes BioGRID-compatible columns (target, feature, fold_change, fdr,
    rank, abs_fold_change) and saves a ranked CSV
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank GSE250558 targeted perturb-seq DE results")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to GSE250558_all_results_combined.tsv.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write ranked CSV",
    )
    parser.add_argument(
        "--drop-guides",
        nargs="*",
        default=["SAFE", "raf.transgene"],
        help="Guide/perturbation labels to drop (controls/driver)",
    )
    return parser.parse_args()


def normalize_guides(series: pd.Series) -> pd.Series:
    """Uppercase and strip whitespace from guide names."""
    return series.astype(str).str.upper().str.strip()


def normalize_gene(series: pd.Series) -> pd.Series:
    """Uppercase and strip whitespace from gene symbols."""
    return series.astype(str).str.upper().str.strip()


def collapse_by_best_hit(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the best row per (target, feature) using lowest adj p-value then largest |lfc|."""
    df = df.copy()
    df["abs_lfc"] = df["lfc"].abs()
    df_sorted = df.sort_values(["target", "feature", "adj_pval", "abs_lfc"], ascending=[True, True, True, False])
    return df_sorted.groupby(["target", "feature"], as_index=False).first()


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank within each target by absolute fold change."""
    df = df.copy()
    df["abs_fold_change"] = df["fold_change"].abs()
    df["rank"] = df.groupby("target")["abs_fold_change"].rank(method="dense", ascending=False).astype(int)
    return df.sort_values(["target", "rank"]).reset_index(drop=True)


def make_ranked(df: pd.DataFrame, drop_guides: List[str]) -> pd.DataFrame:
    drop_upper = {g.upper() for g in drop_guides}

    df = df.copy()
    df["target"] = normalize_guides(df["guide"])
    df["feature"] = normalize_gene(df["name"])
    df["fold_change"] = df["lfc"]
    df["fdr"] = df["adj_pval"]
    df["p_value"] = df["pval"]
    df["source_time"] = df["time"]
    df["source_index"] = df["index"]

    df = df[~df["target"].isin(drop_upper)].copy()

    collapsed = collapse_by_best_hit(df)
    ranked = add_ranks(collapsed)

    expected_targets = ranked["target"].nunique()
    print(f"Targets after dropping controls: {expected_targets}")
    print(f"Target labels: {', '.join(sorted(ranked['target'].unique()))}")
    print(f"Total rows: {len(ranked)}")

    columns = [
        "target",
        "feature",
        "fold_change",
        "fdr",
        "p_value",
        "abs_fold_change",
        "rank",
        "source_time",
        "source_index",
    ]
    return ranked[columns]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, sep="\t")

    ranked = make_ranked(df, args.drop_guides)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)
    print(f"Saved ranked data to {args.output}")


if __name__ == "__main__":
    main()
