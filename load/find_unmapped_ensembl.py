#!/usr/bin/env python3
"""
Find Ensembl IDs that were not mapped to gene symbols during conversion.
"""

import pandas as pd
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Find Ensembl IDs that were not mapped to gene symbols'
    )
    parser.add_argument('original_csv', help='Path to original CSV file with Ensembl IDs')
    parser.add_argument('converted_csv', help='Path to converted CSV file with gene symbols')
    parser.add_argument('--output', help='Path to save unmapped Ensembl IDs (optional)')
    
    args = parser.parse_args()
    
    # Load original CSV
    logger.info(f"Loading original CSV from {args.original_csv}")
    df_original = pd.read_csv(args.original_csv)
    logger.info(f"Original CSV has {len(df_original)} rows")
    
    # Load converted CSV
    logger.info(f"Loading converted CSV from {args.converted_csv}")
    df_converted = pd.read_csv(args.converted_csv)
    logger.info(f"Converted CSV has {len(df_converted)} rows")
    
    # Get Ensembl IDs from original (feature column)
    original_ensembl_ids = set(df_original['feature'].unique())
    logger.info(f"Found {len(original_ensembl_ids)} unique Ensembl IDs in original CSV")
    
    # Get Ensembl IDs from converted (ensembl_id column if it exists, otherwise feature)
    if 'ensembl_id' in df_converted.columns:
        converted_ensembl_ids = set(df_converted['ensembl_id'].unique())
        logger.info("Using 'ensembl_id' column from converted CSV")
    else:
        # If no ensembl_id column, check if feature column still has Ensembl IDs
        # (this shouldn't happen if conversion worked, but let's check)
        converted_ensembl_ids = set()
        logger.warning("No 'ensembl_id' column found in converted CSV")
    
    # Find unmapped Ensembl IDs
    unmapped_ensembl_ids = original_ensembl_ids - converted_ensembl_ids
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total unique Ensembl IDs in original: {len(original_ensembl_ids)}")
    logger.info(f"Total unique Ensembl IDs in converted: {len(converted_ensembl_ids)}")
    logger.info(f"Unmapped Ensembl IDs: {len(unmapped_ensembl_ids)}")
    
    if len(unmapped_ensembl_ids) > 0:
        logger.info(f"\nUnmapped Ensembl IDs ({len(unmapped_ensembl_ids)}):")
        unmapped_sorted = sorted(unmapped_ensembl_ids)
        
        # Show first 50
        for i, ensembl_id in enumerate(unmapped_sorted[:50]):
            logger.info(f"  {ensembl_id}")
        
        if len(unmapped_sorted) > 50:
            logger.info(f"  ... and {len(unmapped_sorted) - 50} more")
        
        # Count rows affected
        rows_with_unmapped = df_original[df_original['feature'].isin(unmapped_ensembl_ids)]
        logger.info(f"\nRows affected: {len(rows_with_unmapped)}")
        
        # Show breakdown by target
        if 'target' in df_original.columns:
            target_counts = rows_with_unmapped['target'].value_counts()
            logger.info(f"\nTop 10 targets affected:")
            for target, count in target_counts.head(10).items():
                logger.info(f"  {target}: {count} rows")
        
        # Save to file if requested
        if args.output:
            logger.info(f"\nSaving unmapped Ensembl IDs to {args.output}")
            unmapped_df = pd.DataFrame({
                'ensembl_id': sorted(unmapped_ensembl_ids)
            })
            unmapped_df.to_csv(args.output, index=False)
            logger.info(f"Saved {len(unmapped_df)} unmapped Ensembl IDs")
            
            # Also save detailed breakdown
            output_path = Path(args.output)
            detailed_output = output_path.parent / f"{output_path.stem}_detailed.csv"
            rows_with_unmapped.to_csv(detailed_output, index=False)
            logger.info(f"Saved detailed breakdown to {detailed_output}")
    else:
        logger.info("\nAll Ensembl IDs were successfully mapped!")
    
    return 0


if __name__ == "__main__":
    exit(main())

