import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from utils.loss_utils import LossClass
from utils.metrics_utils import concordance_index


def notebook_compare():
    """Exercise the actual comparison function maintained in the notebook."""
    path = Path(__file__).resolve().parents[1] / 'notebook/compare_model_performance_genomics.ipynb'
    notebook = json.loads(path.read_text())
    nodes = [node for cell in notebook['cells'] if cell['cell_type'] == 'code'
             for node in ast.parse(''.join(cell['source'])).body
             if isinstance(node, ast.FunctionDef) and node.name == 'compare']
    assert len(nodes) == 1
    namespace = dict(np=np, pd=pd, Path=Path, FOLD=0, concordance_index=concordance_index)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['compare']


def test_comparison_rejects_different_patients(tmp_path):
    roots = [tmp_path/'without', tmp_path/'with']
    for root, mode in zip(roots, ['without_mutation', 'mutation']):
        root.mkdir(parents=True)
        pd.DataFrame({'case_id': ['A', 'B'],
            'label': [0, 1], 'prediction': [0, 1], 'censorship': [0, 1],
            'event_time': [1., 2.], 'risk': [.8, .2], 'hazard_0': [.6, .2],
            'hazard_1': [.3, .5]}).to_csv(root/f'split_resmlp_genes.csv_0_{mode}_predictions.csv', index=False)
    paths = [roots[0]/'split_resmlp_genes.csv_0_without_mutation_predictions.csv',
             roots[1]/'split_resmlp_genes.csv_0_mutation_predictions.csv']
    compare = notebook_compare()
    result = compare(*paths)
    assert len(result) == 2 and result.val_cindex.eq(1).all()
    logits = torch.logit(torch.tensor([[.6, .3], [.2, .5]], dtype=torch.float64))
    expected = LossClass()(logits, torch.tensor([0, 1]), torch.tensor([0., 1.])).item()
    assert result.val_survival_nll.to_numpy() == pytest.approx([expected, expected])
    path = paths[1]
    frame = pd.read_csv(path)
    frame.loc[0, 'case_id'] = 'C'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='patients'):
        compare(*paths)
