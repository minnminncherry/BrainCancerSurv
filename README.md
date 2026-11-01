# Brain Cancer Survival Prediction

A research project investigating survival prediction for glioblastoma (GBM) and lower-grade glioma (LGG) using TCGA clinical records, gene-expression profiles, whole-slide pathology images (WSI), and MRI.

The research compares individual data modalities with genomic–WSI fusion to explore whether complementary molecular and tissue information improves prediction. Mutation-data processing and biological exploration are also included; mutation-aware modeling is an ongoing research direction rather than a completed component of every model.

## Research approach

Clinical records provide survival times and censoring information. Most experiments convert survival times into four quantile-based classes, ordered from shorter to longer survival. Models output class logits, and a derived risk score is used to calculate the concordance index (C-index).

The current training objectives are cross-entropy or sigmoid binary cross-entropy with one-hot class targets. These are survival-time classification objectives; they do not directly optimize a censor-aware survival likelihood. Censoring is used when calculating C-index.

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

The genomic ResMLP script includes a smaller model, dropout, weight decay, and patient-grouped inner monitoring for early stopping. It restores the checkpoint with the best monitoring loss before evaluating the outer held-out fold. These controls reduce overfitting risk but do not guarantee improved generalization.

## Results and visualization

Training writes:

- `result/<model>/<dataset>/summary.csv`: per-fold training and held-out loss/C-index, plus mean results.
- `result/<model>/<dataset>/training_history_fold_<fold>.csv`: epoch histories; early-stopping runs also include monitoring loss and the selected epoch.
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
