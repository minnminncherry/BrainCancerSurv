"""Save reproducible ResMLP inference bundles and evaluate an external cohort."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.gene_resnet_mlp_model import ResMLPGenomics
from utils.loss_utils import LossClass, survival_outputs
from utils.metrics_utils import concordance_index


def save_genomic_bundle(args, model, train_dataset, fold, output_dir):
    factory = args.data_factory
    scaler = factory.scaler_
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    mode = 'mutation' if model.mutation_head is not None else 'without_mutation'
    id_col = next(c for c in factory.ID_CANDIDATES if c in train_dataset.df)
    bundle = dict(state_dict=state, source_file=args.genomic_file_name, fold=fold,
        loss_func=args.loss_func, label_col=args.label_col,
        feature_cols=list(scaler['feature_cols']), mean=scaler['mean'].tolist(),
        std=scaler['std'].tolist(), bin_edges=factory.survival_bin_edges.tolist(),
        training_patient_ids=train_dataset.df[id_col].astype(str).unique().tolist(),
        mutation_genes=list(getattr(factory, 'mutation_gene_cols', [])) if model.mutation_head is not None else [],
        model_config=dict(input_dim=model.input_dim, n_classes=model.n_classes,
            hidden_dim=model.input_projection[1].out_features,
            num_blocks=len(model.residual_blocks), dropout=model.input_projection[3].p,
            n_mutation_targets=model.mutation_head.out_features if model.mutation_head is not None else 0))
    path = Path(output_dir)/f'genomic_fold_{fold}_{mode}_bundle.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return path


def evaluate_external(bundle_path, expression_path, clinical_path, output_path):
    bundle = torch.load(bundle_path, map_location='cpu', weights_only=True)
    if str(bundle['loss_func']).lower() not in {'nll_surv','nllsurvloss','nll_survival','nll_loss','lossclass'}:
        raise ValueError('External survival evaluation requires an NLL-trained model.')
    expression = pd.read_csv(expression_path)
    clinical = pd.read_csv(clinical_path)
    candidates = ('patient_id', '_PATIENT', 'sampleID', 'bcr_patient_barcode')
    for frame in (expression, clinical):
        col = next(c for c in candidates if c in frame)
        frame.rename(columns={col:'patient_id'}, inplace=True)
        frame.dropna(subset=['patient_id'], inplace=True)
        frame['patient_id'] = frame.patient_id.astype(str).str.strip()
        frame.drop_duplicates('patient_id', inplace=True)
    features = bundle['feature_cols']
    missing = set(features) - set(expression)
    if missing:
        raise ValueError(f'External expression lacks {len(missing)} LGG training genes; use a compatible expression matrix.')
    label_col = bundle['label_col']
    clinical[label_col] = pd.to_numeric(clinical[label_col], errors='coerce')
    status_col = next(c for c in ('CDE_vital_status','vital_status') if c in clinical)
    clinical['censorship'] = clinical[status_col].astype(str).str.strip().str.upper().map(
        {'LIVING':1., 'ALIVE':1., 'DECEASED':0., 'DEAD':0.})
    if 'days_to_death' in clinical:
        clinical.loc[pd.to_numeric(clinical.days_to_death, errors='coerce').notna(), 'censorship'] = 0.
    clinical = clinical.loc[np.isfinite(clinical[label_col]) & clinical[label_col].ge(0) & clinical.censorship.notna()]
    frame = clinical[['patient_id',label_col,'censorship']].merge(expression[['patient_id',*features]], on='patient_id')
    if frame.empty:
        raise ValueError('No eligible external patients.')
    if set(frame.patient_id) & set(bundle['training_patient_ids']):
        raise ValueError('External patients overlap the source training cohort.')
    x = frame[features].apply(pd.to_numeric, errors='raise').to_numpy(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('External expression must contain finite gene values.')
    x = (x - np.asarray(bundle['mean'],np.float32)) / np.asarray(bundle['std'],np.float32)
    # Freeze source-cohort boundaries; never fit bins or scaling on GBM.
    edges = np.asarray(bundle['bin_edges'],float)
    labels = np.searchsorted(edges[1:-1],frame[label_col].to_numpy(),side='left')
    model = ResMLPGenomics(**bundle['model_config']).eval()
    model.load_state_dict(bundle['state_dict'])
    logits, mutation_probs = [], []
    with torch.no_grad():
        for batch in torch.tensor(x).split(64):
            output = model(batch)
            if isinstance(output,dict):
                mutation_probs.append(output['genomic_mutation_logits'].sigmoid())
                output = output['survival_logits']
            logits.append(output)
    logits = torch.cat(logits)
    hazards, survival, probs = survival_outputs(logits)
    risk = -survival.sum(1).numpy()
    predictions = probs.argmax(1).numpy()
    loss = LossClass()(logits,torch.tensor(labels),torch.tensor(frame.censorship.to_numpy())).item()
    output = frame[['patient_id',label_col,'censorship']].rename(columns={'patient_id':'case_id',label_col:'event_time'})
    output = output.assign(label=labels,prediction=predictions,risk=risk)
    probabilities = {f'hazard_{i}': hazards[:,i].numpy() for i in range(hazards.shape[1])}
    probabilities.update({f'prob_class_{i}': probs[:,i].numpy() for i in range(probs.shape[1])})
    if mutation_probs:
        values = torch.cat(mutation_probs).numpy()
        probabilities.update({f'genomic_mutation_prob_{gene}':values[:,i] for i,gene in enumerate(bundle['mutation_genes'])})
    output = pd.concat([output.reset_index(drop=True),pd.DataFrame(probabilities)],axis=1)
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    output.to_csv(output_path,index=False)
    observed = frame.censorship.eq(0).to_numpy()
    return dict(patients=len(frame), external_cindex=concordance_index(frame[label_col],frame.censorship,risk),
        external_survival_nll=loss, observed_bin_accuracy=float((predictions[observed]==labels[observed]).mean())
        if observed.any() else np.nan)
