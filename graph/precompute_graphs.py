"""
Precompute compact graph JSON for the ontology mini-graph UI.

Outputs live next to the raw deconvoluted data:
/data/dhruvgautam/gbm_perturb/deconvoluted/graphs/<dataset>_graph.json

Run: python precompute_graphs.py
Requires write access to /data/dhruvgautam/gbm_perturb/deconvoluted/graphs.
"""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path("/data/dhruvgautam/gbm_perturb/deconvoluted")
OUT = SRC / "graphs"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for path in sorted(SRC.glob("*_deconvoluted.json")):
        data = json.loads(path.read_text())
        graphs = {}
        perturbations = []

        for pert, entry in data.items():
            deconv = entry.get("deconvoluted") or []
            if not deconv:
                continue
            perturbations.append(pert)

            fc_map = {}
            for item in deconv:
                gene = item["gene"]
                fc = item["fold_change"]
                prev = fc_map.get(gene)
                if prev is None or abs(fc) > abs(prev["fold_change"]):
                    fc_map[gene] = {"gene": gene, "fold_change": fc}

            ontology = set()
            for field in ("reactome", "bp", "cc", "mf"):
                block = entry.get(field) or {}
                for key in ("in_hop1", "not_in_hop1"):
                    for obj in block.get(key) or []:
                        ontology.add(obj["gene"])

            edges = []
            for gene, obj in fc_map.items():
                if gene == pert:
                    continue
                edges.append(
                    {
                        "source": pert,
                        "target": gene,
                        "fold_change": obj["fold_change"],
                        "known": gene in ontology,
                    }
                )

            nodes = [pert] + [g for g in fc_map.keys() if g != pert]
            graphs[pert] = {"nodes": nodes, "edges": edges}

        out = {
            "file": path.name,
            "perturbations": sorted(perturbations),
            "graphs": graphs,
        }
        out_path = OUT / path.name.replace("_deconvoluted.json", "_graph.json")
        out_path.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=True))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
