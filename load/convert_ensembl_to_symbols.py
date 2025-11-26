#!/usr/bin/env python3
"""
Convert Ensembl IDs in the feature column of a ranked CSV to gene symbols.
Unmapped Ensembl IDs are kept and used as-is (they won't match graph annotations,
so they'll be considered "not in hop 1" during deconvolution).
"""

import pandas as pd
import argparse
import logging
import sys
from pathlib import Path

# Import conversion functions from ranked.py
sys.path.insert(0, str(Path(__file__).parent))
from ranked import map_ensembl_to_symbols, strip_ensembl_version

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_ensembl_to_symbols_keep_unmapped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Ensembl IDs in the feature column to gene symbols.
    Unmapped Ensembl IDs are kept and used as-is (they won't match graph annotations).
    
    Args:
        df: DataFrame with DE data containing 'feature' column with Ensembl IDs
        
    Returns:
        DataFrame with feature column converted to gene symbols (or Ensembl ID if unmapped)
    """
    logger.info("Converting Ensembl IDs to gene symbols (keeping unmapped)")
    
    df_converted = df.copy()
    
    # Get unique Ensembl IDs
    ensembl_ids = df_converted['feature'].unique().tolist()
    
    # Get mapping
    id_to_symbol = map_ensembl_to_symbols(ensembl_ids)
    
    # Apply mapping - use Ensembl ID as fallback for unmapped genes
    df_converted['gene_symbol'] = df_converted['feature'].map(id_to_symbol)
    
    # For unmapped genes, use the Ensembl ID itself (without version) as the identifier
    # This ensures they're kept but won't match anything in GO/Reactome graphs
    unmapped_mask = df_converted['gene_symbol'].isna()
    unmapped_count = unmapped_mask.sum()
    
    if unmapped_count > 0:
        logger.info(f"Found {unmapped_count} features that could not be mapped to gene symbols")
        logger.info("Keeping unmapped genes - they will be considered 'not in hop 1' during deconvolution")
        # Use Ensembl ID (without version) as the identifier for unmapped genes
        df_converted.loc[unmapped_mask, 'gene_symbol'] = df_converted.loc[unmapped_mask, 'feature'].apply(
            lambda x: strip_ensembl_version(str(x))
        )
    
    # Store original Ensembl ID
    df_converted['ensembl_id'] = df_converted['feature']
    
    # Replace feature column with gene symbols (or Ensembl ID for unmapped)
    df_converted['feature'] = df_converted['gene_symbol']
    df_converted = df_converted.drop(columns=['gene_symbol'])
    
    mapped_count = len(df_converted) - unmapped_count
    unique_genes = df_converted['feature'].nunique()
    logger.info(f"Converted {mapped_count} IDs to gene symbols, kept {unmapped_count} unmapped as Ensembl IDs")
    logger.info(f"Total unique features: {unique_genes}")
    
    return df_converted


def main():
    parser = argparse.ArgumentParser(
        description='Convert Ensembl IDs in feature column to gene symbols (keeping unmapped)'
    )
    parser.add_argument('input_csv', help='Path to input CSV file with Ensembl IDs in feature column')
    parser.add_argument('output_csv', help='Path to output CSV file with gene symbols')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_csv).exists():
        logger.error(f"Input file not found: {args.input_csv}")
        return 1
    
    logger.info(f"Loading CSV from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    logger.info(f"Loaded {len(df)} rows")
    
    # Convert Ensembl IDs to gene symbols (keeping unmapped)
    df_converted = convert_ensembl_to_symbols_keep_unmapped(df)
    
    # Save converted CSV
    logger.info(f"Saving converted CSV to {args.output_csv}")
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_converted.to_csv(args.output_csv, index=False)
    logger.info(f"Saved {len(df_converted)} rows to {args.output_csv}")
    
    logger.info("Conversion completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())

