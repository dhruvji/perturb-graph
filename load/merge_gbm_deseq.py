#!/usr/bin/env python3
"""
Merge DESeq output CSVs from GBM perturb-seq experiments by experimental group.

Groups:
  - invitro_RT: In vitro with radiotherapy
  - invitro_noRT: In vitro without radiotherapy
  - preinf_RT: Pre-infected with radiotherapy
  - preinf_noRT: Pre-infected without radiotherapy
  - CED_RT: Convection Enhanced Delivery with radiotherapy
  - CED_noRT: Convection Enhanced Delivery without radiotherapy
  - Astro: Astrocytes (tumor microenvironment)
  - Macrophages: Macrophages (tumor microenvironment)
  - Microglia: Microglia (tumor microenvironment)
  - OPC: Oligodendrocyte precursor cells (tumor microenvironment)
  - Oligo: Oligodendrocytes (tumor microenvironment)
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define the experimental groups and their file patterns
GROUPS = {
    # GL261 in vitro experiments
    "invitro_RT": {"prefix": "invitro_", "suffix": "_RT_non-targeting_RT.csv"},
    "invitro_noRT": {"prefix": "invitro_", "suffix": "_noRT_non-targeting_noRT.csv"},
    
    # GL261 pre-infected experiments
    "preinf_RT": {"prefix": "preinf_", "suffix": "_RT_non-targeting_RT.csv"},
    "preinf_noRT": {"prefix": "preinf_", "suffix": "_noRT_non-targeting_noRT.csv"},
    
    # GL261 CED experiments
    "CED_RT": {"prefix": "CED_", "suffix": "_RT_non-targeting_RT.csv"},
    "CED_noRT": {"prefix": "CED_", "suffix": "_noRT_non-targeting_noRT.csv"},
    
    # SB28 tumor microenvironment experiments
    "Astro": {"prefix": "Astro_", "suffix": "_sgNegCtrl_3.csv"},
    "Macrophages": {"prefix": "Macrophages_", "suffix": "_sgNegCtrl_3.csv"},
    "Microglia": {"prefix": "Microglia_", "suffix": "_sgNegCtrl_3.csv"},
    "OPC": {"prefix": "OPC_", "suffix": "_sgNegCtrl_3.csv"},
    "Oligo": {"prefix": "Oligo_", "suffix": "_sgNegCtrl_3.csv"},
}


def extract_target_gene(filename: str, prefix: str, suffix: str) -> str:
    """Extract the target gene name from the filename."""
    # Remove prefix and suffix to get target gene
    name = filename.replace(prefix, "").replace(suffix, "")
    return name


def load_and_format_deseq(filepath: Path, target_gene: str) -> pd.DataFrame:
    """
    Load a DESeq output CSV and format it for build_graphs.py.
    
    The DESeq output has columns:
    - feature: gene symbol
    - log_fc: log fold change
    - padj: adjusted p-value (FDR)
    
    We convert to the format expected by build_graphs.py:
    - target: the perturbed gene
    - feature: the measured gene
    - fold_change: log fold change
    - fdr: adjusted p-value
    """
    logger.info(f"Loading {filepath}")
    
    # These CSVs use space as delimiter (R-style output)
    df = pd.read_csv(filepath, sep=r'\s+')
    
    # The CSV uses R-style indexing, first column might be unnamed row index
    if df.columns[0] == 'Unnamed: 0' or str(df.columns[0]).isdigit():
        df = df.drop(columns=[df.columns[0]])
    
    # Rename columns to expected format
    df_formatted = pd.DataFrame({
        'target': target_gene,
        'feature': df['feature'],
        'fold_change': df['log_fc'],
        'fdr': df['padj'],
    })
    
    # Remove rows with missing values
    df_formatted = df_formatted.dropna(subset=['feature', 'fold_change', 'fdr'])
    
    logger.info(f"  Loaded {len(df_formatted)} genes for target {target_gene}")
    
    return df_formatted


def merge_group_csvs(
    input_dir: Path,
    group_name: str,
    output_path: Path
) -> pd.DataFrame:
    """
    Merge all DESeq CSVs for a specific experimental group.
    
    Args:
        input_dir: Directory containing DESeq CSV files
        group_name: Name of the experimental group
        output_path: Path to save merged CSV
        
    Returns:
        Merged DataFrame
    """
    if group_name not in GROUPS:
        raise ValueError(f"Unknown group: {group_name}. Valid groups: {list(GROUPS.keys())}")
    
    group_config = GROUPS[group_name]
    prefix = group_config["prefix"]
    suffix = group_config["suffix"]
    
    logger.info(f"Merging group: {group_name}")
    logger.info(f"  Prefix: {prefix}, Suffix: {suffix}")
    
    # Find all matching files
    pattern = f"{prefix}*{suffix}"
    matching_files = list(input_dir.glob(pattern))
    
    if not matching_files:
        logger.warning(f"No files found matching pattern: {pattern}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(matching_files)} files")
    
    # Load and merge all files
    dfs = []
    for filepath in sorted(matching_files):
        target_gene = extract_target_gene(filepath.name, prefix, suffix)
        df = load_and_format_deseq(filepath, target_gene)
        dfs.append(df)
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"\nMerged dataset statistics:")
    logger.info(f"  Total rows: {len(merged_df)}")
    logger.info(f"  Unique targets: {merged_df['target'].nunique()}")
    logger.info(f"  Unique features: {merged_df['feature'].nunique()}")
    
    # Save merged CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Saved merged CSV to {output_path}")
    
    return merged_df


def rank_by_fold_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank genes by absolute fold change within each perturbation.
    """
    logger.info("Ranking by absolute fold change")
    
    df = df.copy()
    df['abs_fold_change'] = np.abs(df['fold_change'])
    
    # Group by target and rank
    df['rank'] = df.groupby('target')['abs_fold_change'].rank(
        method='dense', ascending=False
    ).astype(int)
    
    # Sort by target and rank
    df_sorted = df.sort_values(['target', 'rank'])
    
    return df_sorted


def main():
    parser = argparse.ArgumentParser(
        description="Merge DESeq CSVs from GBM perturb-seq experiments by experimental group",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available groups:
{chr(10).join(f'  - {g}' for g in GROUPS.keys())}

Example usage:
  python merge_gbm_deseq.py --input_dir /path/to/deseq_output --group invitro_RT --output /path/to/output.csv
  python merge_gbm_deseq.py --input_dir /path/to/deseq_output --all --output_dir /path/to/output/
        """
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/home/dhruvgautam/gbm_perturb/shared_data/deseq_output"),
        help="Directory containing DESeq CSV files"
    )
    parser.add_argument(
        "--group",
        type=str,
        choices=list(GROUPS.keys()),
        help="Experimental group to merge"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all groups"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (for single group)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/data/dhruvgautam/gbm_perturb/merged"),
        help="Output directory (for --all mode)"
    )
    parser.add_argument(
        "--fdr_threshold",
        type=float,
        default=None,
        help="Optional FDR threshold for filtering (default: no filtering)"
    )
    parser.add_argument(
        "--add_ranks",
        action="store_true",
        help="Add rank column based on absolute fold change"
    )
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
    
    if args.all:
        # Process all groups
        for group_name in GROUPS.keys():
            output_path = args.output_dir / f"{group_name}_merged.csv"
            df = merge_group_csvs(args.input_dir, group_name, output_path)
            
            if len(df) > 0:
                if args.fdr_threshold is not None:
                    df = df[df['fdr'] <= args.fdr_threshold]
                    logger.info(f"After FDR filtering ({args.fdr_threshold}): {len(df)} rows")
                
                if args.add_ranks:
                    df = rank_by_fold_change(df)
                    df.to_csv(output_path, index=False)
                    logger.info(f"Added ranks and saved to {output_path}")
            
            print()  # Empty line between groups
    else:
        if not args.group:
            logger.error("Must specify either --group or --all")
            return 1
        
        if not args.output:
            args.output = args.output_dir / f"{args.group}_merged.csv"
        
        df = merge_group_csvs(args.input_dir, args.group, args.output)
        
        if len(df) > 0:
            if args.fdr_threshold is not None:
                df = df[df['fdr'] <= args.fdr_threshold]
                logger.info(f"After FDR filtering ({args.fdr_threshold}): {len(df)} rows")
                df.to_csv(args.output, index=False)
            
            if args.add_ranks:
                df = rank_by_fold_change(df)
                df.to_csv(args.output, index=False)
                logger.info(f"Added ranks and saved to {args.output}")
    
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
