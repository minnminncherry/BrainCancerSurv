from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


def _split_genes(raw: str) -> List[str]:
    genes: List[str] = []
    for gene in raw.split(","):
        gene = gene.strip()
        if gene:
            genes.append(gene)
    return genes


def _parse_hallmark_line(line: str) -> Optional[Tuple[str, List[str]]]:
    """
    Parse a single line that defines one hallmark pathway.

    Supported common formats:
      1) "HALLMARK_XYZ ABCA1,ABCB8,ACAA2"
      2) "HALLMARK_XYZ,ABCA1,ABCB8,ACAA2"
      3) Tab-separated: "HALLMARK_XYZ\tABCA1,ABCB8,ACAA2" or "HALLMARK_XYZ\tABCA1\tABCB8\tACAA2"
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if "\t" in stripped:
        parts = [p.strip() for p in stripped.split("\t") if p.strip()]
        if len(parts) < 2:
            return None
        pathway = parts[0]
        if len(parts) == 2 and "," in parts[1]:
            genes = _split_genes(parts[1])
        else:
            # Some files store genes as separate tab columns.
            genes = parts[1:]
        return pathway, genes

    if "," in stripped and (" " not in stripped.split(",", 1)[0]):
        # Looks like a CSV list (pathway, gene1, gene2, ...)
        tokens = next(csv.reader([stripped]))
        tokens = [t.strip() for t in tokens if t.strip()]
        if len(tokens) < 2:
            return None
        pathway = tokens[0]
        genes = tokens[1:]
        return pathway, genes

    # Fallback: split into "PATHWAY <comma-separated genes...>".
    parts = stripped.split(None, 1)
    if len(parts) != 2:
        return None
    pathway, genes_raw = parts[0].strip(), parts[1].strip()
    genes = _split_genes(genes_raw) if "," in genes_raw else [g for g in genes_raw.split() if g]
    return pathway, genes


def hallmark_pathway_row_to_column_matrix(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    gene_column_name: str = "gene",
    fill_value: int = 0,
    on_value: int = 1,
    dtype: str = "int8",
) -> pd.DataFrame:
    """
    Convert a "row-based" hallmark pathway file into a "column-based" 0/1 matrix.

    Input: one pathway per row (name + genes).
    Output: rows are genes, columns are pathways, values are 0/1 membership.

    Returns the matrix as a pandas DataFrame. If output_path is provided, writes CSV.
    """
    input_path = Path(input_path)
    pathway_to_genes: "OrderedDict[str, List[str]]" = OrderedDict()

    for line in input_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = _parse_hallmark_line(line)
        if parsed is None:
            continue
        pathway, genes = parsed
        if not pathway:
            continue
        # De-duplicate genes per pathway while preserving order.
        seen = set()
        unique_genes: List[str] = []
        for gene in genes:
            gene = gene.strip()
            if not gene or gene in seen:
                continue
            seen.add(gene)
            unique_genes.append(gene)
        pathway_to_genes[pathway] = unique_genes

    all_genes = sorted({g for genes in pathway_to_genes.values() for g in genes})
    pathways = list(pathway_to_genes.keys())

    matrix = pd.DataFrame(fill_value, index=all_genes, columns=pathways, dtype=dtype)
    for pathway, genes in pathway_to_genes.items():
        present = [g for g in genes if g in matrix.index]
        if present:
            matrix.loc[present, pathway] = on_value

    matrix.index.name = gene_column_name

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(output_path, index=True)

    return matrix


if __name__ == "__main__":
    import yaml
    import os

    yaml_file_path = os.path.join(os.getcwd(), "../config.yaml")
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    default_input = data['pathway_src_file_path']['hallmark_pathway']
    default_output = data['pathway_clean_file_path']['hallmark_pathway_matrix_output_file']

    hallmark_pathway_row_to_column_matrix(default_input, output_path=default_output)
