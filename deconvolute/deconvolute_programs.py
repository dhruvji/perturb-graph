#!/usr/bin/env python3
"""
Program-level enrichment analysis for GBM perturb-seq.

For each perturbation, this script looks at differentially expressed genes
(filtered by FDR) and summarizes how much absolute fold-change signal falls
into each predefined transcriptional program (gbm_programs.csv).

Outputs:
  - CSV with one row per perturbation/program containing coverage and
    absolute-effect ratios.
  - JSON with per-perturbation details and hit gene lists.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import pandas as pd


def load_program_definitions(programs_csv: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load program definitions from CSV.

    Returns a mapping:
        program_name -> {
            "category": str,
            "genes": set[str]
        }
    """
    if not programs_csv.exists():
        raise FileNotFoundError(f"Programs CSV not found: {programs_csv}")

    df = pd.read_csv(programs_csv)
    required_cols = {"program_name", "program_category", "gene_symbol"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Programs CSV missing columns: {missing}")

    programs: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        name = str(row["program_name"]).strip()
        category = str(row["program_category"]).strip()
        gene = str(row["gene_symbol"]).strip().upper()

        if not name or not gene:
            continue

        if name not in programs:
            programs[name] = {"category": category, "genes": set()}
        programs[name]["genes"].add(gene)

    logging.info(f"Loaded {len(programs)} programs from {programs_csv}")
    return programs


def load_de_results(csv_path: Path, fdr_threshold: float) -> pd.DataFrame:
    """
    Load merged DE results and filter by FDR.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"target", "feature", "fold_change", "fdr"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    df = df.dropna(subset=["target", "feature", "fold_change", "fdr"]).copy()
    df = df[df["fdr"] < fdr_threshold].copy()

    # Normalize to uppercase symbols for matching to program genes.
    df["target_upper"] = df["target"].astype(str).str.upper().str.strip()
    df["feature_upper"] = df["feature"].astype(str).str.upper().str.strip()
    df["abs_fold_change"] = df["fold_change"].abs()

    logging.info(
        "Loaded %d DE rows passing FDR < %.3f across %d perturbations",
        len(df),
        fdr_threshold,
        df["target_upper"].nunique(),
    )
    return df


def summarize_program_for_target(
    target_df: pd.DataFrame,
    target_gene: str,
    programs: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Compute per-program metrics for a single perturbation.
    """
    total_abs_fc = target_df["abs_fold_change"].sum()
    program_rows: List[Dict[str, Any]] = []
    program_json: Dict[str, Any] = {}

    for prog_name, prog in programs.items():
        prog_genes: Set[str] = prog["genes"]
        hits_df = target_df[target_df["feature_upper"].isin(prog_genes)]

        num_hits = len(hits_df)
        abs_fc_sum = float(hits_df["abs_fold_change"].sum()) if num_hits else 0.0
        abs_effect_ratio = abs_fc_sum / total_abs_fc if total_abs_fc > 0 else 0.0
        coverage = num_hits / len(prog_genes) if prog_genes else 0.0
        mean_abs_fc = float(hits_df["abs_fold_change"].mean()) if num_hits else 0.0
        max_abs_fc = float(hits_df["abs_fold_change"].max()) if num_hits else 0.0

        program_rows.append(
            {
                "target": target_gene,
                "program_name": prog_name,
                "program_category": prog["category"],
                "num_program_genes": len(prog_genes),
                "num_hits": num_hits,
                "coverage": coverage,
                "abs_fold_change_sum": abs_fc_sum,
                "abs_effect_ratio": abs_effect_ratio,
                "mean_abs_fold_change": mean_abs_fc,
                "max_abs_fold_change": max_abs_fc,
                "total_abs_fold_change_target": total_abs_fc,
            }
        )

        program_json[prog_name] = {
            "category": prog["category"],
            "num_program_genes": len(prog_genes),
            "num_hits": num_hits,
            "coverage": coverage,
            "abs_fold_change_sum": abs_fc_sum,
            "abs_effect_ratio": abs_effect_ratio,
            "mean_abs_fold_change": mean_abs_fc,
            "max_abs_fold_change": max_abs_fc,
            "genes_hit": [
                {
                    "gene": row["feature_upper"],
                    "fold_change": float(row["fold_change"]),
                    "abs_fold_change": float(row["abs_fold_change"]),
                }
                for _, row in hits_df.sort_values(
                    "abs_fold_change", ascending=False
                ).iterrows()
            ],
        }

    return program_rows, {
        "total_abs_fold_change": total_abs_fc,
        "programs": program_json,
    }


def compute_program_enrichment(
    df: pd.DataFrame,
    programs: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute program enrichment metrics for all perturbations.
    """
    csv_rows: List[Dict[str, Any]] = []
    json_output: Dict[str, Any] = {}

    for target_gene, target_df in df.groupby("target_upper"):
        rows, target_json = summarize_program_for_target(
            target_df, target_gene, programs
        )
        csv_rows.extend(rows)
        json_output[target_gene] = target_json

    csv_df = pd.DataFrame(csv_rows)
    logging.info(
        "Computed program metrics for %d perturbations (%d rows)",
        len(json_output),
        len(csv_df),
    )
    return csv_df, json_output


def save_outputs(csv_df: pd.DataFrame, json_data: Dict[str, Any], csv_path: Path, json_path: Path) -> None:
    """Save CSV and JSON outputs."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    csv_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    logging.info("Saved CSV to %s", csv_path)
    logging.info("Saved JSON to %s", json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize program-level absolute fold-change effects per perturbation"
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Merged DE results CSV (columns: target, feature, fold_change, fdr)",
    )
    parser.add_argument(
        "--programs_csv",
        type=Path,
        default=Path(__file__).resolve().parent / "gbm_programs.csv",
        help="Program definitions CSV (default: gbm_programs.csv in this directory)",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Path to write program enrichment table (CSV)",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Path to write per-perturbation JSON details",
    )
    parser.add_argument(
        "--fdr_threshold",
        type=float,
        default=0.05,
        help="FDR cutoff to define DE genes (default: 0.05)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    programs = load_program_definitions(args.programs_csv)
    df = load_de_results(args.input_csv, args.fdr_threshold)
    csv_df, json_data = compute_program_enrichment(df, programs)
    save_outputs(csv_df, json_data, args.output_csv, args.output_json)
    logging.info("Done!")


if __name__ == "__main__":
    main()
