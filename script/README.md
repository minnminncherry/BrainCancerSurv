# Training scripts

All training scripts use the same argument order:

```text
script.sh CANCER MODEL LOSS
```

Examples:

```bash
sh script/genomic.sh GBM resmlp nll_surv
sh script/wsi.sh LGG mil cross_entropy
sh script/mri.sh GBM ViT nll_surv
sh script/fusion_genomic_wsi.sh GBM multimodal_cross_attention nll_surv
sh script/fusion_genomic_wsi_bothdataset.sh gbm_lgg multimodal_late_fusion nll_surv
sh script/fusion_genomic_mri.sh LGG multimodal_mri_late_fusion cross_entropy
```

Use `nll_surv`, `cross_entropy`, or `sigmoid` for the loss. Use `GBM` or `LGG`
for the cohort; genomic and WSI scripts also support `gbm_lgg` or `both`.
Combined MRI metadata is not prepared.

Each script retains its own training settings. Extra training options can follow
the three arguments, for example `--epoch 30 --batch_size 2`.
Set `PYTHON_BIN` to your environment's Python executable if needed.

Collect the commands you want in `run_model.sh`, following its existing example,
and run:

```bash
sh script/run_model.sh
```

`model_run.txt` is a reference list of all 48 experiments, updated to the same
argument order. `run_model.sh` executes only the commands written inside it.
`mutation.sh` remains empty because there is no mutation training entry point.
No training or cron installation is performed by editing these scripts.

## Genomic + WSI + MRI together

These models match patients who have all three modalities and clinical labels.
The new entry point reuses the existing training loop without changing `main.py`.

```bash
sh script/fusion_genomic_wsi_mri.sh GBM multimodal_wsi_mri_late_fusion nll_surv
sh script/fusion_genomic_wsi_mri.sh LGG multimodal_wsi_mri_cross_attention nll_surv
```

Late fusion concatenates ResMLP, WSI MIL, and MRI ViT embeddings before the
survival head. Cross-attention uses one genomic query for two attention branches:
WSI patch tokens and 343 ViT MRI tokens (for 224³ volumes). Their outputs are concatenated with
the genomic embedding before the survival head. Both return raw survival logits;
`NLLSurvLoss` converts them into hazards. Cross-attention can also return WSI
attention weights and a 7×7×7 MRI attention map (for 224³ volumes).

All MRI scans from a patient stay in the same fold. Each sample uses one MRI
scan, the patient's expression vector, and concatenated patches from all listed
WSI slides. Scans therefore repeat the patient's survival outcome. WSI PT files
must contain a `[patches, features]` tensor, or a dictionary with a `features`
tensor. The selected WSI feature dimension must match the stored features.
The cross-attention branch follows the existing WSI model's unmasked-token
behavior; the launcher defaults to batch size 1 to avoid collate padding.

Default hidden size is 32, with one ResMLP block. Use the same matched patients
for comparisons against the two-modality models. More modalities do not
guarantee better validation performance, particularly with a small matched cohort.
Existing genomic–WSI and genomic–MRI checkpoints are not automatically loaded;
these joint models train together from initialization.

MRI fusion now uses `ViTMRI` (32³ patches, hidden size 128, two Transformer
layers, four heads), followed by a projection to the fusion hidden size.
Late fusion uses attention-pooled features; cross-attention keeps patch tokens.
CNN-based MRI fusion checkpoints are incompatible with this encoder change.


## Updated diagram: mutation-informed survival

```bash
sh script/fusion_genomic_wsi_mri.sh GBM mutation_guided_cross_attention nll_surv
sh script/fusion_genomic_wsi_mri.sh GBM mutation_guided_late_fusion nll_surv
```

The diagram model uses ResMLP, ViT MRI features, and WSI MIL. Separate MRI/WSI
mutation heads predict the genes selected by `--mutation_genes`; separate embedding
heads return `[B, 64]` by default. Both embeddings and the predicted mutation
probabilities feed the survival head. True mutation labels supervise the mutation
heads only. Missing mutation labels are masked, not converted to negative labels.

The script exposes `--embedding_dim 64`, `--resmlp_hidden_dim 64`,
`--resmlp_num_blocks 1`, `--attention_heads 4`, `--mri_hidden_dim 128`,
`--mri_num_layers 2`, `--mri_patch_size 32`, `--mri_mutation_weight 1.0`, and
`--wsi_mutation_weight 1.0`. Extra command-line options at the end override defaults.
For example, append `--embedding_dim 128` or `--mutation_genes TP53 EGFR`.

The default mutation CSV is `dataset_csv/genomic_data/filter_gene_<cohort>_mutation.csv`.
GBM has this file; LGG requires a matching LGG mutation CSV supplied via
`MUTATION_FILE=/absolute/path/to/lgg_mutations.csv`. Mutation tables contain
`patient_id` and 0/1 gene columns; missing entries are allowed. A recorded zero
means no recorded mutation, not verified assay coverage.

Saved `train_loss` and `final_val_loss` include survival loss plus both weighted
mutation BCE terms. C-index is computed from survival predictions. Prediction CSVs
add `mri_mutation_prob_<gene>` and `wsi_mutation_prob_<gene>`. Result pickle files
also store each sample's MRI/WSI embeddings (existing patient-keyed storage keeps
the last scan for a repeated patient). `run_config.json` records the script arguments.

Legacy survival-only model names and comparison models remain available. Existing
three-modality names passed to this script are mapped to the new model names when
`--mutation_file` is supplied, giving separate result directories. These are new
models and require training; old survival-only checkpoints are incompatible.
`main.py` is unchanged.


Mutation-file parsing is centralized in `dataset/SurvivalMutationDataset.py`:
`_load_mutation_file()` validates one shared CSV and gene order, and
`_prepare_mutation_labels()` attaches labels/masks to split metadata.
The expression-only `SurvivalMutationDataset` uses the same reader.
The three-modality dataset now loads inputs only; pass `--mutation_file` to the
training entry point rather than its dataset constructor.

Each supervised run saves `mutation_genes.csv` and
`mutation_label_coverage_fold_<fold>.csv` under its model/dataset result directory.
Coverage is counted by unique patient. Missing labels are masked and are not
negative examples. One CSV can cover multiple modalities/cohorts only when patient
IDs and gene columns match; use a common held-out patient set for fair comparisons.
Existing survival-only baselines do not acquire mutation prediction heads merely
by supplying `--mutation_file`.

TensorBoard logging is implemented in `utils/tensorboard_utils.py` with small
hooks in `core_utils.py`. Install requirements, run training normally, then use:

```bash
tensorboard --logdir runs
```

Open the local URL printed by TensorBoard. Runs are separated by model, dataset,
mutation mode, fold, and a unique session ID. `TENSORBOARD_LOG_DIR` overrides the
log root. Training curves include C-index, observed-event bin accuracy, survival
loss, weighted mutation-loss contribution, total loss, and learning rate.
Validation metrics are final points after training, not per-epoch curves.
C-index and observed-event bin accuracy measure different aspects of performance.
Total loss includes mutation supervision when enabled. Existing CSV results are
unchanged; older runs are not automatically imported into TensorBoard.

Genomic survival/mutation execution flow:

1. `genomic.sh` selects expression, clinical and optional mutation files.
2. `SurvivalGenomicDataset` matches expression/clinical patients and creates
   patient splits; its expression scaler is fitted on training patients only.
3. Before model creation, `core_utils._train_val` calls
   `SurvivalMutationDataset._prepare_mutation_labels`. This reads the mutation
   file, fixes target-gene order and attaches per-patient labels and masks.
4. `_init_model` sizes ResMLP's optional mutation head from those target genes.
5. Gene expression enters ResMLP. Survival and mutation labels supervise their
   respective losses; the sum updates the shared encoder during training.
6. Validation runs without weight updates and saves survival predictions and,
   when enabled, mutation probabilities in the mode-specific prediction CSV.

Without a mutation file, mutation preparation clears any old target schema and
masks, and ResMLP keeps its survival-only output. `SurvivalMutationDataset` itself
also supports an independent expression-to-mutation dataset; it is not used as
the survival dataset factory in `genomic.sh`.
