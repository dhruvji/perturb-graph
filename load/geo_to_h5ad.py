#!/usr/bin/env python3
"""
Convert GEO count matrix files (especially featureCounts output) to h5ad format.
"""
import argparse
import gzip
import re
import pandas as pd
import numpy as np
from scipy import sparse
import anndata as ad
from pathlib import Path


def peek_file(filepath: str, n_lines: int = 20) -> list[str]:
    """Peek at the first n lines of a file to understand its structure."""
    opener = gzip.open if filepath.endswith('.gz') else open
    mode = 'rt' if filepath.endswith('.gz') else 'r'
    lines = []
    with opener(filepath, mode) as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            lines.append(line.rstrip('\n'))
    return lines


def read_featurecounts(filepath: str) -> pd.DataFrame:
    """
    Read featureCounts output format.
    Format: comment line, then header (Geneid, Chr, Start, End, Strand, Length, sample1, sample2, ...)
    """
    print(f"Reading featureCounts file: {filepath}")
    
    # Count comment lines (lines starting with #)
    lines = peek_file(filepath, 10)
    skip_rows = 0
    for line in lines:
        if line.startswith('#'):
            skip_rows += 1
        else:
            break
    
    print(f"Skipping {skip_rows} comment line(s)")
    
    # Read the file, skipping comment lines
    df = pd.read_csv(
        filepath,
        sep='\t',
        skiprows=skip_rows,
        index_col=0,  # Geneid column
        compression='gzip' if filepath.endswith('.gz') else None,
    )
    
    print(f"Initial shape: {df.shape}")
    print(f"Index name: {df.index.name}")
    print(f"Columns: {list(df.columns[:10])}...")
    
    # featureCounts has annotation columns: Chr, Start, End, Strand, Length
    # Drop these to keep only count columns
    annotation_cols = ['Chr', 'Start', 'End', 'Strand', 'Length']
    cols_to_drop = [c for c in annotation_cols if c in df.columns]
    if cols_to_drop:
        print(f"Dropping featureCounts annotation columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Clean up sample names (remove path and .bam/.dedup.bam suffix)
    def clean_sample_name(name: str) -> str:
        # Extract just the filename from path
        name = Path(name).stem
        # Remove common suffixes
        name = re.sub(r'\.dedup$', '', name)
        name = re.sub(r'\.sorted$', '', name)
        return name
    
    df.columns = [clean_sample_name(c) for c in df.columns]
    print(f"Cleaned sample names: {list(df.columns[:5])}...")
    
    # Ensure all values are numeric
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    
    print(f"Final matrix shape (genes x samples): {df.shape}")
    return df


def counts_to_anndata(df: pd.DataFrame, accession: str) -> ad.AnnData:
    """Convert counts DataFrame to AnnData object."""
    # GEO/featureCounts: genes in rows, samples in columns
    # AnnData expects: samples (obs) x genes (var)
    X = sparse.csr_matrix(df.T.values.astype(np.float32))
    
    adata = ad.AnnData(X=X)
    adata.obs_names = df.columns.astype(str)  # samples
    adata.var_names = df.index.astype(str)    # genes
    
    # Store raw counts in a layer
    adata.layers["counts"] = adata.X.copy()
    adata.uns["geo_accession"] = accession
    
    # Parse sample metadata from sample names (Donor_X_Gene format)
    sample_pattern = re.compile(r'(Donor_\d+)_(.+?)(?:_\d+)?$')
    
    donors = []
    targets = []
    for name in adata.obs_names:
        match = sample_pattern.match(name)
        if match:
            donors.append(match.group(1))
            target = match.group(2)
            # Consolidate all AAVS1 variants (AAVS1_1, AAVS1_2, etc.) into single "AAVS1" control
            if target.startswith('AAVS1'):
                target = 'AAVS1'
            targets.append(target)
        else:
            donors.append('unknown')
            targets.append(name)
    
    adata.obs['donor'] = donors
    adata.obs['target_gene'] = targets
    adata.obs['is_control'] = adata.obs['target_gene'] == 'AAVS1'
    
    print(f"AnnData object: {adata.n_obs} samples x {adata.n_vars} genes")
    print(f"Unique donors: {adata.obs['donor'].nunique()}")
    print(f"Unique targets: {adata.obs['target_gene'].nunique()}")
    
    return adata


def main():
    parser = argparse.ArgumentParser(description="Convert GEO/featureCounts to h5ad")
    parser.add_argument("--input", "-i", required=True, help="Input counts file (can be .gz)")
    parser.add_argument("--output", "-o", required=True, help="Output h5ad file")
    parser.add_argument("--accession", "-a", default="unknown", help="GEO accession ID")
    args = parser.parse_args()
    
    # Read counts (featureCounts format)
    df = read_featurecounts(args.input)
    
    if df.empty:
        raise ValueError("No data found in counts file. Check file format.")
    
    # Convert to AnnData
    adata = counts_to_anndata(df, args.accession)
    
    # Write output
    print(f"Writing AnnData to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    adata.write(args.output, compression="gzip")
    print("Done.")


if __name__ == "__main__":
    main()
