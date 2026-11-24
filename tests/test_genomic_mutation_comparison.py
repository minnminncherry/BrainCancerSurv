import numpy as np
import pandas as pd
import pytest
import torch

from script.compare_genomic_mutation import compare, metrics
from utils.loss_utils import LossClass


def test_survival_nll_matches_training_loss():
    logits = torch.tensor([[.2, -.5], [-.4, .7]], dtype=torch.float64)
    frame = pd.DataFrame({'hazard_0': logits.sigmoid()[:, 0], 'hazard_1': logits.sigmoid()[:, 1],
        'label': [1, 0], 'prediction': [1, 0], 'censorship': [0, 1],
        'event_time': [1., 2.], 'risk': [.8, .2]})
    expected = LossClass()(logits, torch.tensor([1, 0]), torch.tensor([0., 1.])).item()
    assert metrics(frame)['val_survival_nll'] == pytest.approx(expected)


def test_comparison_rejects_different_patients(tmp_path):
    roots = [tmp_path/'without', tmp_path/'with']
    for root, mode in zip(roots, ['without_mutation', 'with_mutation']):
        folder = root/'resmlp/genes.csv'
        folder.mkdir(parents=True)
        np.save(folder/'survival_bin_edges_fold_0.npy', [0, 1, 2])
        pd.DataFrame({'case_id': ['A', 'B'], 'mutation_supervision': mode,
            'label': [0, 1], 'prediction': [0, 1], 'censorship': [0, 1],
            'event_time': [1., 2.], 'risk': [.8, .2], 'hazard_0': [.6, .2],
            'hazard_1': [.3, .5]}).to_csv(root/'split_resmlp_genes.csv_0_predictions.csv', index=False)
    result = compare(*roots, 'resmlp', 'genes.csv')
    assert len(result) == 2 and result.val_cindex.eq(1).all()
    path = roots[1]/'split_resmlp_genes.csv_0_predictions.csv'
    frame = pd.read_csv(path)
    frame.loc[0, 'case_id'] = 'C'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='patient IDs'):
        compare(*roots, 'resmlp', 'genes.csv')
