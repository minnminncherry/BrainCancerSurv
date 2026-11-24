# Brain Cancer Survival Prediction

A research project investigating survival prediction for glioblastoma (GBM) and lower-grade glioma (LGG) using TCGA clinical records, gene-expression profiles, whole-slide pathology images (WSI), and MRI.

The research compares individual data modalities with genomic–WSI fusion to explore whether complementary molecular and tissue information improves prediction. Mutation-data processing and biological exploration are also included; mutation-aware modeling is an ongoing research direction rather than a completed component of every model.

## Research approach

Clinical records provide survival times and censoring information. Most experiments convert survival times into four quantile-based classes, ordered from shorter to longer survival. Models output class logits, and a derived risk score is used to calculate the concordance index (C-index).

Training supports cross-entropy, sigmoid binary cross-entropy with one-hot targets, and `NLLSurvLoss` (`--loss_func nll_surv`). The first two are classification objectives. NLLSurvLoss optimizes a discrete-time likelihood that uses censoring labels (1 = censored, 0 = observed death); its logits represent interval hazards. Censoring is also used when calculating C-index.

| Input | Models | Purpose |
|---|---|---|
| Gene expression | MLP, SNN, Gen2Vec, ResMLP | Learn molecular representations for survival classification |
| WSI patch features | Attention-based MIL, TransMIL | Aggregate patch representations into a slide representation |
| 3D MRI | Vision Transformer | Learn spatial relationships between volumetric patches |
| Gene expression + WSI | Late fusion, cross-attention | Combine molecular and pathology information |

MRI is currently evaluated as a separate branch. The implemented multimodal fusion models combine genomics and WSI, not all three modalities together.

## Model overview

### Genomic–WSI late fusion

ResMLP produces a genomic embedding, while attention-based MIL produces a pooled WSI embedding. Their concatenation is projected into a fused representation before classification.

```text
Gene expression → ResMLP → 256 features ──┐
                                        ├→ Concatenate (512) → Fusion (256) → Survival classes
WSI patches → Attention MIL → 256 features┘
```

These dimensions describe the default fusion configuration. Despite its name, this implementation fuses learned features before the final classifier.

### Genomic–WSI cross-attention

A ResMLP genomic embedding forms the query. MIL attention-weighted WSI patch tokens form the keys and values. Cross-attention generates a genomic-guided WSI representation, which is refined by a feed-forward network and concatenated with the original genomic embedding for classification.

The current cross-attention implementation does not mask zero-padded WSI tokens and does not include residual connections around the fusion attention/feed-forward blocks. Standalone MIL and late fusion retain MIL's default padding masking.

### MRI Transformer

The MRI branch loads NIfTI volumes, applies canonical orientation and foreground Z-score normalization, and center-crops or pads to `224 × 224 × 224` voxels. The default model uses `32³` patches, 128-dimensional embeddings, two Transformer encoder layers, and four attention heads. Attention pooling feeds a four-class classifier.

Each MRI sequence is currently processed separately. Reference tumor masks in visualization notebooks are annotations, not segmentation outputs from the survival model. MRI metadata should contain the intended image inputs.

## Repository structure

```text
BrainCanerSurvival/
├── main.py                 # Cross-validation experiment entry point
├── config.yaml             # Local data and preprocessing configuration
├── requirements.txt        # Python dependencies
├── clean_dataset/          # Clinical, genomic, mutation, WSI, and MRI preparation
├── dataset/                # Dataset loaders and split construction
├── dataset_csv/            # Clinical labels, metadata, genomic tables, pathways
├── model/                  # Individual models and fusion architectures
├── utils/                  # Arguments, training, losses, evaluation, batching
├── script/                 # Example experiment commands
├── notebook/               # Analysis and visualization notebooks
├── tests/                  # Small model tests with synthetic inputs
└── result/                 # Summaries, predictions, histories, checkpoints, figures
```

## Installation

Use a Python environment compatible with the pinned dependencies. CI currently uses Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CPU-only execution, install CPU PyTorch first, then the remaining requirements:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU execution, select the appropriate PyTorch wheel for the CUDA environment. Let PyTorch resolve its matching CUDA dependencies rather than copying NVIDIA package pins from a local environment.

Some preprocessing and notebook workflows additionally require packages such as `jupyterlab`, `nbformat`, `seaborn`, `scikit-learn`, `scipy`, `PyMySQL`, `SQLAlchemy`, `h5py`, `opencv-python`, and `openslide-python`. OpenSlide also requires its native library. Install the packages needed for the workflow you run; the main requirements file is not a complete specification of every exploratory notebook environment.

## Data preparation

Before running an experiment:

1. Prepare clinical records and modality metadata under `dataset_csv/`.
2. Match records using patient identifiers and check for duplicate IDs and missing survival labels.
3. For WSI experiments, prepare patch-feature `.pt` files and point `PT_DIR` to their location.
4. For MRI experiments, prepare NIfTI files and metadata with patient IDs, filenames, and folder paths.
5. Update `config.yaml` and the paths in the selected shell script. Training scripts have their own paths; changing the configuration alone does not update them.

Raw images and source-data directories are local resources and are not downloaded automatically. Some cleaning scripts use MySQL and can replace table contents or export files, so run the relevant preparation steps deliberately.

Fusion samples require matching clinical, genomic, and WSI records. Their count can therefore differ from each individual dataset. Matched row counts should not be interpreted as unique patient counts without checking duplicate IDs.

## Run experiments

Run commands from the repository root after checking the paths and arguments in each script:

```bash
bash script/genomic.sh   # Genomic ResMLP experiment
bash script/wsi.sh       # Attention-based WSI MIL experiment
bash script/mri.sh       # MRI Transformer experiment
bash script/fusion.sh    # Genomic–WSI late fusion experiment
```

To run cross-attention fusion, change the modality in `script/fusion.sh`:

```bash
--modality "multimodal_cross_attention"
```

Use `--n_folds` to control cross-validation; the default is five folds. MRI currently has an explicit four-fold setting in its script. Check the scripts for the currently selected GBM/LGG cohort and optimizer settings.

The genomic ResMLP script includes a smaller model, dropout, and weight decay. Training runs for the configured number of epochs and evaluates the final checkpoint on the held-out fold. These controls reduce overfitting risk but do not guarantee improved generalization.

## Results and visualization

Training writes:

- `result/<model>/<dataset>/summary.csv`: per-fold training and held-out loss/C-index, plus mean results.
- `result/<model>/<dataset>/training_history_fold_<fold>.csv`: epoch histories and the final selected epoch.
- `result/split_<model>_<dataset>_<fold>_predictions.csv`: held-out class predictions, probabilities, and risk scores.
- `result/model_checkpoints/`: saved model weights and prediction artifacts.
- `result/comparisons/`: exported figures and comparison tables.

Checkpoint and result filenames can be reused by later runs. Archive a completed run before starting another experiment with the same model/dataset settings. Old checkpoints are not necessarily compatible with revised architectures.

Open the analysis notebooks with:

```bash
jupyter lab notebook/
```

| Notebook | Analysis |
|---|---|
| [Genomic visualization](notebook/genomic_data_visualization.ipynb) | Model comparisons, ResMLP confusion matrices, training/validation curves, and runtime plots |
| [Multimodal analysis](notebook/multimodal_analysis.ipynb) | Late-fusion and cross-attention results |
| [MRI visualization](notebook/mri_data_visualization.ipynb) | Predictions, patient split previews, and reference-mask comparisons |
| [WSI visualization](notebook/wsi_data_visualization.ipynb) | Tissue masks and patch extraction previews |
| [Genomic cleaning](notebook/genomic_data_clean.ipynb) | Exploratory preparation and optional baseline experiments |
| [Biological exploration](notebook/learning_biological.ipynb) | Notes and exploration related to mutation and gene information |

See [notebook/README.md](notebook/README.md) for notebook-specific requirements and execution notes.

## Evaluation and reproducibility

Report mean and variation across folds for C-index and loss. Accuracy and confusion matrices describe survival-class classification and should be distinguished from survival ranking. When reporting runtime, use comparable hardware, data splits, and training budgets.

For fair comparisons, verify patient separation, duplicate handling, preprocessing, and survival-bin boundaries for each dataset implementation. MRI currently fits survival bins on training patients; other dataset loaders should be checked independently. Saved historical results may reflect earlier architecture or preprocessing versions.

## Continuous integration

The GitHub Actions workflow installs CPU PyTorch, checks dependency consistency and Python syntax, and runs synthetic model tests without requiring patient data or MySQL.

```bash
python -m pip install 'pytest>=8,<9'
python -m pytest tests -q
```

### Censor-aware late-fusion training

`script/fusion.sh` uses `--loss_func nll_surv`. ResMLP and attention MIL produce normalized embeddings, which are concatenated and projected before predicting hazard logits. For multimodal experiments, eligible clinical/genomic/WSI patients are matched and deduplicated before fold construction. Time-bin boundaries for NLL are fitted using observed events in the training fold and saved as `survival_bin_edges_fold_<fold>.npy`.

NLL prediction CSVs include hazards, survival curves, and `survival_tail_probability`. For compatibility, `prob_class_*` columns contain interval event masses; these sum to `1 - survival_tail_probability`, not necessarily 1. C-index risk is the negative sum of predicted survival probabilities. `correct` reports label/prediction equality for every row. `event_bin_observed` is false for censored follow-up bins, which are not known death classes; use this flag or censorship to restrict event-bin accuracy to observed deaths. Loss magnitudes from NLL and cross-entropy should not be compared directly. Retrain after switching objectives; existing classification checkpoints and historical plots do not establish survival-model performance.

## Mutation prediction dataset and head

`SurvivalMutationDataset` prepares a separate multi-label mutation task: gene-expression values are inputs, and recorded binary mutation flags are targets. Clinical survival labels are not required. This dictionary-based dataset is not registered with the survival-only `main.py` loop.

```python
import torch
from torch.utils.data import DataLoader
from dataset.SurvivalMutationDataset import SurvivalMutationDataset
from model.mutation_head import MutationPredictionHead

data = SurvivalMutationDataset(
    "dataset_csv/genomic_data/normalized_zscore_gbm.csv",
    "dataset_csv/genomic_data/filter_gene_gbm_mutation.csv",
    target_genes=["IDH1", "TP53", "EGFR", "PTEN"],
)
train, validation, scaler = data.return_splits(validation_fraction=0.2)
loader = DataLoader(train, batch_size=16, shuffle=True)
head = MutationPredictionHead(data.input_dim, data.n_targets)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train.positive_weights())
optimizer = torch.optim.Adam(head.parameters(), lr=1e-4, weight_decay=1e-4)

# One training step; repeat over batches/epochs for an experiment.
batch = next(iter(loader))
optimizer.zero_grad()
logits = head(batch["expression"])
loss = criterion(logits, batch["mutation"])
loss.backward()
optimizer.step()
```

The head can alternatively accept a learned genomic embedding by setting `input_dim` to that embedding's dimension. At inference, apply `logits.sigmoid()` for one probability per target gene in `data.mutation_gene_cols` order; softmax is inappropriate because multiple genes can be mutated together.

Patients are matched by ID, duplicates are rejected, and missing mutation labels are not converted to negatives. Additional scaling is fitted on training patients only, although pre-existing normalization in source CSVs remains. Inspect `train.target_statistics()` for genes lacking either mutation class. A zero target means no recorded mutation, not confirmed assay coverage. This preparation and the one-step example do not establish mutation-prediction performance.
