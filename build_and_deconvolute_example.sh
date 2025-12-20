#!/bin/bash
#SBATCH --cpus-per-task=64
#SBATCH --mem=512GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/dhruvgautam/logs/build_and_deconvolute_${ACCESSION_NUMBER}/build_and_deconvolute_${ACCESSION_NUMBER}_%j.out
#SBATCH --error=/home/dhruvgautam/logs/build_and_deconvolute_${ACCESSION_NUMBER}/build_and_deconvolute_${ACCESSION_NUMBER}_%j.err
#SBATCH --job-name=build_deconv_${ACCESSION_NUMBER}
#SBATCH --ntasks=1
#SBATCH --nodes=1

ACCESSION_NUMBER={}

RAW_COMBINED=/data/dhruvgautam/geo_extract/${ACCESSION_NUMBER}/${ACCESSION_NUMBER}_all_results_combined.tsv.gz
RANKED_CSV=/data/dhruvgautam/geo_extract/${ACCESSION_NUMBER}/de_results/${ACCESSION_NUMBER}_ranked_symbols.csv
GRAPHS_OUTPUT=/data/dhruvgautam/geo_extract/${ACCESSION_NUMBER}/de_results/go_graphs.json
DECONVOLUTED_OUTPUT=/data/dhruvgautam/geo_extract/${ACCESSION_NUMBER}/de_results/deconvoluted_genes_20_0.05.json
GO_DIR=/data/dhruvgautam/go # save gene ontology jsons here

mkdir -p /data/dhruvgautam/geo_extract/${ACCESSION_NUMBER}/de_results

echo "=========================================="
echo "Step 0: Compute DE for all perturbations"
echo "=========================================="
python /home/dhruvgautam/perturb-graph/load/prepare_${ACCESSION_NUMBER}_ranked.py \ 
  --input "${RAW_COMBINED}" \
  --output "${RANKED_CSV}" 

if [ $? -ne 0 ]; then
    echo "Error: prepare_${ACCESSION_NUMBER}_ranked.py failed"
    exit 1
fi

echo "=========================================="
echo "Step 1: Building GO/Reactome graphs from ranked CSV"
echo "=========================================="
python /home/dhruvgautam/perturb-graph/build/build_graphs.py \
  --input_csv "${RANKED_CSV}" \
  --output "${GRAPHS_OUTPUT}" \
  --go_dir "${GO_DIR}" \
  --max_hops 10 \
  --num_workers 64

if [ $? -ne 0 ]; then
    echo "Error: build_graphs.py failed"
    exit 1
fi

echo "=========================================="
echo "Step 2: Deconvoluting GO/Reactome graphs"
echo "=========================================="
python /home/dhruvgautam/perturb-graph/deconvolute/deconvolute_graphs.py \
  --input_csv "${RANKED_CSV}" \
  --graphs_json "${GRAPHS_OUTPUT}" \
  --output "${DECONVOLUTED_OUTPUT}" \
  --fdr_threshold 0.05 \
  --top_n 20

if [ $? -ne 0 ]; then
    echo "Error: deconvolute_graphs.py failed"
    exit 1
fi

echo "=========================================="
echo "All GO/Reactome steps completed successfully!"
echo "Graphs: ${GRAPHS_OUTPUT}"
echo "Deconvoluted: ${DECONVOLUTED_OUTPUT}"
echo "=========================================="
