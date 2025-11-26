#!/usr/bin/env python3
"""
Script to create sorted ranked gene lists from differential expression data.

This script processes DE data by:
1. Loading DE results CSV with Ensembl IDs
2. Converting Ensembl IDs to gene symbols using mygene
3. Filtering by FDR threshold (optional, default 0.05)
4. Ranking by fold_change (absolute value)
5. Saving sorted ranked data
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging
from typing import Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_de_data(filepath: str) -> pd.DataFrame:
    """
    Load differential expression data from CSV file.
    
    Args:
        filepath: Path to the CSV file containing DE data
        
    Returns:
        DataFrame with DE data
    """
    logger.info(f"Loading DE data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def strip_ensembl_version(ensembl_id: str) -> str:
    """Strip version number from Ensembl ID (e.g., ENSG00000223972.5 -> ENSG00000223972)."""
    if pd.isna(ensembl_id):
        return ensembl_id
    return str(ensembl_id).split('.')[0]


def map_ensembl_to_symbols(ensembl_ids: list) -> Dict[str, str]:
    """
    Map Ensembl IDs to gene symbols using mygene.
    
    Args:
        ensembl_ids: List of Ensembl IDs (with or without version numbers)
        
    Returns:
        Dictionary mapping Ensembl IDs to gene symbols
    """
    try:
        import mygene
    except ImportError:
        logger.error("mygene package not installed. Install with: pip install mygene")
        raise
    
    logger.info(f"Mapping {len(ensembl_ids)} Ensembl IDs to gene symbols")
    
    # Strip version numbers for querying
    stripped_ids = [strip_ensembl_version(eid) for eid in ensembl_ids]
    unique_stripped = list(set(stripped_ids))
    
    logger.info(f"Found {len(unique_stripped)} unique Ensembl IDs (after stripping versions)")
    
    # Query mygene
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        unique_stripped, 
        scopes='ensembl.gene', 
        fields='symbol', 
        species='human',
        verbose=False
    )
    
    # Build mapping from stripped ID to symbol
    stripped_to_symbol = {}
    for result in results:
        query_id = result.get('query')
        symbol = result.get('symbol')
        if query_id and symbol:
            stripped_to_symbol[query_id] = symbol
    
    # Build mapping from original ID (with version) to symbol
    original_to_symbol = {}
    for orig_id in ensembl_ids:
        stripped = strip_ensembl_version(orig_id)
        if stripped in stripped_to_symbol:
            original_to_symbol[orig_id] = stripped_to_symbol[stripped]
    
    mapped_count = len(original_to_symbol)
    unmapped_count = len(ensembl_ids) - mapped_count
    logger.info(f"Successfully mapped {mapped_count} IDs, {unmapped_count} unmapped")
    
    return original_to_symbol


def convert_ensembl_to_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Ensembl IDs in the feature column to gene symbols.
    
    Args:
        df: DataFrame with DE data containing 'feature' column with Ensembl IDs
        
    Returns:
        DataFrame with feature column converted to gene symbols
    """
    logger.info("Converting Ensembl IDs to gene symbols")
    
    df_converted = df.copy()
    
    # Get unique Ensembl IDs
    ensembl_ids = df_converted['feature'].unique().tolist()
    
    # Get mapping
    id_to_symbol = map_ensembl_to_symbols(ensembl_ids)
    
    # Apply mapping
    df_converted['gene_symbol'] = df_converted['feature'].map(id_to_symbol)
    
    # Check for unmapped features
    unmapped = df_converted['gene_symbol'].isna().sum()
    if unmapped > 0:
        logger.warning(f"Found {unmapped} features that could not be mapped to gene symbols")
        df_converted = df_converted.dropna(subset=['gene_symbol'])
        logger.info(f"Removed {unmapped} unmapped features, {len(df_converted)} rows remaining")
    
    # Replace feature column with gene symbols
    df_converted['ensembl_id'] = df_converted['feature']
    df_converted['feature'] = df_converted['gene_symbol']
    df_converted = df_converted.drop(columns=['gene_symbol'])
    
    unique_genes = df_converted['feature'].nunique()
    logger.info(f"Converted to {unique_genes} unique gene symbols")
    
    return df_converted


def filter_by_fdr(df: pd.DataFrame, fdr_threshold: float = 0.05) -> pd.DataFrame:
    """
    Filter data by FDR threshold.
    
    Args:
        df: DataFrame with DE data
        fdr_threshold: FDR threshold for filtering (default 0.05)
        
    Returns:
        Filtered DataFrame
    """
    logger.info(f"Filtering by FDR threshold: {fdr_threshold}")
    filtered_df = df[df['fdr'] <= fdr_threshold].copy()
    logger.info(f"After FDR filtering: {len(filtered_df)} rows (from {len(df)})")
    return filtered_df


def rank_by_fold_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank genes by absolute fold change within each perturbation.
    
    Args:
        df: DataFrame with DE data
        
    Returns:
        DataFrame with ranking information added, sorted by target and rank
    """
    logger.info("Ranking by absolute fold change")
    
    # Calculate absolute fold change for ranking
    df = df.copy()
    df['abs_fold_change'] = np.abs(df['fold_change'])
    
    # Group by target (perturbation) and rank by absolute fold change
    df['rank'] = df.groupby('target')['abs_fold_change'].rank(method='dense', ascending=False)
    
    # Sort by target and rank
    df_sorted = df.sort_values(['target', 'rank']).copy()
    
    # Convert rank to integer
    df_sorted['rank'] = df_sorted['rank'].astype(int)
    
    logger.info(f"Ranking complete. Found {df_sorted['target'].nunique()} unique perturbations")
    return df_sorted


def save_ranked_data(df: pd.DataFrame, output_path: str):
    """
    Save ranked data to CSV file.
    
    Args:
        df: DataFrame with ranked data
        output_path: Path to save the output file
    """
    logger.info(f"Saving ranked data to {output_path}")
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics for the ranked data."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total perturbations: {df['target'].nunique()}")
    print(f"Total gene entries: {len(df)}")
    print(f"Unique genes: {df['feature'].nunique()}")
    print(f"Mean genes per perturbation: {len(df) / df['target'].nunique():.1f}")
    
    # Show top 10 perturbations by number of genes
    pert_counts = df['target'].value_counts().head(10)
    print(f"\nTop 10 perturbations by gene count:")
    for pert, count in pert_counts.items():
        print(f"  {pert}: {count} genes")
    
    # Show sample of ranked output
    print(f"\nSample output (first perturbation, top 5 genes):")
    first_target = df['target'].iloc[0]
    sample = df[df['target'] == first_target].head(5)
    print(sample[['target', 'feature', 'fold_change', 'fdr', 'rank']].to_string(index=False))
    
    print("="*60)


def create_ranked_de(
    input_file: str,
    output_file: str,
    fdr_threshold: float = 0.05,
    apply_fdr_filter: bool = True,
    skip_ensembl_mapping: bool = False
) -> pd.DataFrame:
    """
    Main function to create ranked DE data from input CSV.
    
    Args:
        input_file: Path to input CSV file with DE data
        output_file: Path to output CSV file for ranked data
        fdr_threshold: FDR threshold for filtering (default 0.05)
        apply_fdr_filter: Whether to filter by FDR threshold (default True)
        skip_ensembl_mapping: Skip Ensembl ID to gene symbol mapping (default False)
        
    Returns:
        DataFrame with ranked data
    """
    # Load data
    df = load_de_data(input_file)
    
    # Convert Ensembl IDs to gene symbols
    if not skip_ensembl_mapping:
        df = convert_ensembl_to_symbols(df)
    
    # Filter by FDR if requested
    if apply_fdr_filter:
        df = filter_by_fdr(df, fdr_threshold)
        if len(df) == 0:
            logger.warning("No data passed FDR filtering threshold")
            return pd.DataFrame()
    
    # Rank by fold change
    ranked_df = rank_by_fold_change(df)
    
    # Save results
    save_ranked_data(ranked_df, output_file)
    
    # Print summary
    print_summary_stats(ranked_df)
    
    return ranked_df


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Create sorted ranked gene lists from DE data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python ranked.py /data/dhruvgautam/geo_extract/de_results/GSE271788_de_results.csv \\
                   /data/dhruvgautam/geo_extract/de_results/GSE271788_ranked.csv
  
  # Skip FDR filtering
  python ranked.py input.csv output.csv --no-fdr-filter
  
  # Use different FDR threshold
  python ranked.py input.csv output.csv --fdr-threshold 0.1
        """
    )
    parser.add_argument('input_file', help='Path to input CSV file with DE data')
    parser.add_argument('output_file', help='Path to output CSV file for ranked data')
    parser.add_argument('--fdr-threshold', type=float, default=0.05,
                        help='FDR threshold for filtering (default: 0.05)')
    parser.add_argument('--no-fdr-filter', action='store_true',
                        help='Skip FDR filtering and use all data')
    parser.add_argument('--skip-ensembl-mapping', action='store_true',
                        help='Skip Ensembl ID to gene symbol mapping')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    # Process the data
    ranked_data = create_ranked_de(
        input_file=args.input_file,
        output_file=args.output_file,
        fdr_threshold=args.fdr_threshold,
        apply_fdr_filter=not args.no_fdr_filter,
        skip_ensembl_mapping=args.skip_ensembl_mapping
    )
    
    if len(ranked_data) > 0:
        logger.info("Processing completed successfully!")
        return 0
    else:
        logger.error("No data was processed. Please check your input and parameters.")
        return 1


if __name__ == "__main__":
    exit(main())

