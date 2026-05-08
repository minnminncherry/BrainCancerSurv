from __future__ import annotations

import pandas as pd
from pathlib import Path


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

    Input: CSV file with columns: pathway, genes (comma-separated in quotes)
    Output: rows are genes, columns are pathways, values are 0/1 membership.

    Returns the matrix as a pandas DataFrame. If output_path is provided, writes CSV.
    """
    # Read the CSV file
    df = pd.read_csv(input_path, header=None, names=['pathway', 'genes'], quotechar='"')
    
    # Split the genes column into lists
    df['genes'] = df['genes'].apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
    
    # Create a dictionary of pathway to genes
    pathway_to_genes = dict(zip(df['pathway'], df['genes']))
    
    # Get all unique genes
    all_genes = sorted(set(g for genes in pathway_to_genes.values() for g in genes))
    
    # Create the matrix
    matrix = pd.DataFrame(fill_value, index=all_genes, columns=list(pathway_to_genes.keys()), dtype=dtype)
    
    # Fill the matrix
    for pathway, genes in pathway_to_genes.items():
        for gene in genes:
            if gene in matrix.index:
                matrix.loc[gene, pathway] = on_value
    
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
