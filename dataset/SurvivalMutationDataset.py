"""Match gene-expression inputs with binary mutation targets by patient ID."""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_mutation_file(filename, target_genes=None):
    """One mutation-label reader for every modality; missing labels remain unknown."""
    frame = pd.read_csv(filename)
    id_col = next((c for c in ("patient_id", "_PATIENT", "bcr_patient_barcode", "sampleID")
                   if c in frame), None)
    if id_col is None or frame[id_col].isna().any():
        raise ValueError("Mutation data requires a nonmissing patient ID column.")
    frame[id_col] = frame[id_col].astype(str).str.strip()
    if frame[id_col].eq("").any() or frame[id_col].duplicated().any():
        raise ValueError("Mutation patient IDs must be nonempty and unique.")
    frame = frame.set_index(id_col).drop(columns="cohort", errors="ignore")
    genes = list(frame.columns if target_genes is None else target_genes)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("Select at least one unique mutation target gene.")
    missing = set(genes) - set(frame.columns)
    if missing:
        raise ValueError(f"Mutation genes missing from the CSV: {sorted(missing)}")
    labels = frame[genes].apply(pd.to_numeric, errors="raise")
    if not (labels.isin([0, 1]) | labels.isna()).all().all():
        raise ValueError("Mutation labels must be 0, 1, or missing.")
    return labels


def _prepare_mutation_labels(args, train_dataset, validation_dataset, fold):
    """After patient splitting, attach supervision before constructing mutation heads."""
    filename = getattr(args, "mutation_file", None)
    if not filename:
        args.data_factory.mutation_gene_cols = []
        for split in (train_dataset, validation_dataset):
            split.df.drop(columns=["mutation_targets", "mutation_mask"], errors="ignore", inplace=True)
        return
    labels = _load_mutation_file(filename, getattr(args, "mutation_genes", None))
    args.data_factory.mutation_gene_cols = labels.columns.tolist()
    coverage = []
    patient_sets = []
    split_exports = {}
    for name, split in (("train", train_dataset), ("validation", validation_dataset)):
        frame = split.df
        id_col = next((c for c in ("patient_id", "_PATIENT", "bcr_patient_barcode", "sampleID")
                       if c in frame), None)
        if id_col is None:
            raise ValueError(f"No patient ID found in the {name} split.")
        ids = frame[id_col].astype(str).str.strip()
        targets = labels.reindex(ids)
        # __getitem__ already returns row.to_dict(), including these supervision fields.
        # The model inputs remain images/expression only; labels never enter the encoder.
        frame["mutation_targets"] = targets.fillna(0).values.tolist()
        frame["mutation_mask"] = targets.notna().values.tolist()
        exported = frame.drop(columns=["mutation_targets", "mutation_mask"]).copy()
        mutation_columns = {}
        for gene in labels.columns:
            mutation_columns[f"mutation_label_{gene}"] = targets[gene].to_numpy()
            mutation_columns[f"mutation_known_{gene}"] = targets[gene].notna().to_numpy()
        mutation_frame = pd.DataFrame(mutation_columns, index=exported.index)
        split_exports[name] = pd.concat([exported, mutation_frame], axis=1)
        unique = labels.reindex(ids.drop_duplicates())
        patient_sets.append(set(unique.index))
        for gene in labels.columns:
            known = int(unique[gene].notna().sum())
            positive = int(unique[gene].eq(1).sum())
            negative = int(unique[gene].eq(0).sum())
            coverage.append({"split": name, "gene": gene, "patients": len(unique),
                             "known_labels": known, "mutated": positive,
                             "no_recorded_mutation": negative,
                             "unknown_labels": len(unique) - known,
                             "has_both_classes": positive > 0 and negative > 0})
        print(f"Mutation labels ({name}): {int(unique.notna().any(axis=1).sum())}/{len(unique)} patients have labels.")
    guided = str(args.modality).startswith("mutation_guided_")
    if guided and patient_sets[0] & patient_sets[1]:
        raise ValueError("Patients overlap between mutation training and validation; use patient-level splits.")
    report = pd.DataFrame(coverage)
    if guided and report.loc[report["split"] == "train", "known_labels"].sum() == 0:
        raise ValueError("No known mutation labels match the training patients.")
    output = os.path.join(os.path.abspath(os.environ.get("RESULT_DIR") or
                          os.path.join(os.path.dirname(__file__), "..", "result")),
                          args.modality, args.genomic_file_name)
    os.makedirs(output, exist_ok=True)
    for name, exported in split_exports.items():
        exported.to_csv(os.path.join(output, f"{name}_merged_split_fold_{fold}.csv"), index=False)
    labels.columns.to_series(index=None).to_csv(os.path.join(output, "mutation_genes.csv"), index=False, header=["gene"])
    report.to_csv(os.path.join(output, f"mutation_label_coverage_fold_{fold}.csv"), index=False)
    print(report.to_string(index=False))


class MutationSplitDataset(Dataset):
    def __init__(self, x, y, patient_ids, feature_names, target_names):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.df = pd.DataFrame({"patient_id": list(patient_ids)})
        self.genomic_feature_cols = feature_names
        self.mutation_gene_cols = target_names

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "expression": torch.from_numpy(self.x[idx]),
            "mutation": torch.from_numpy(self.y[idx]),
            "patient_id": self.df.iloc[idx]["patient_id"],
        }

    def target_statistics(self):
        positive = self.y.sum(axis=0).astype(int)
        negative = len(self) - positive
        return pd.DataFrame({
            "gene": self.mutation_gene_cols,
            "mutated": positive,
            "no_recorded_mutation": negative,
            "has_both_classes": (positive > 0) & (negative > 0),
        })

    def positive_weights(self):
        """Calculate BCE weights from the training split only."""
        positive = self.y.sum(axis=0)
        negative = len(self) - positive
        has_both = (positive > 0) & (negative > 0)
        return torch.tensor(np.where(has_both, negative / np.maximum(positive, 1), 1),
                            dtype=torch.float32)


class SurvivalMutationDataset:
    """Expression predicts mutation flags; no clinical survival labels are needed."""

    def __init__(self, genomic_file, mutation_file, target_genes=None, seed=42):
        expression = self._read_table(genomic_file)
        mutation = _load_mutation_file(mutation_file, target_genes)
        patients = expression.index.intersection(mutation.index, sort=False)
        self.mutation_gene_cols = list(mutation.columns if target_genes is None else target_genes)
        self.genomic_feature_cols = expression.columns.tolist()
        self.metadata = pd.DataFrame({"patient_id": patients})
        self.x = expression.loc[patients].to_numpy(dtype=np.float32)
        self.y = mutation.loc[patients, self.mutation_gene_cols].to_numpy(dtype=np.float32)
        self.input_dim = len(self.genomic_feature_cols)
        self.n_targets = len(self.mutation_gene_cols)
        self.seed = seed
        if not len(patients) or not self.input_dim or not self.n_targets:
            raise ValueError("Matched patients, expression features, and mutation targets are required.")
        if len(set(self.mutation_gene_cols)) != self.n_targets:
            raise ValueError("Mutation target genes must be unique.")
        if not np.isfinite(self.x).all() or not np.isin(self.y, [0, 1]).all():
            raise ValueError("Expression must be finite and mutation targets must be binary 0/1.")

    @staticmethod
    def _read_table(filename):
        frame = pd.read_csv(filename)
        id_col = next(c for c in ["patient_id", "_PATIENT", "bcr_patient_barcode", "sampleID"] if c in frame)
        if frame[id_col].isna().any():
            raise ValueError("Patient IDs must not be missing.")
        frame[id_col] = frame[id_col].astype(str).str.strip()
        if frame[id_col].eq("").any() or frame[id_col].duplicated().any():
            raise ValueError("Patient IDs must be nonempty and unique.")
        return frame.set_index(id_col).drop(columns="cohort", errors="ignore")

    def __len__(self):
        return len(self.metadata)

    def return_splits(self, fold_indices=None, validation_fraction=0.2, scale=True):
        """Split patients and fit scaling on training expression only."""
        if fold_indices is None:
            if not 0 < validation_fraction < 1:
                raise ValueError("validation_fraction must be between 0 and 1.")
            count = max(1, int(np.ceil(len(self) * validation_fraction)))
            fold_indices = np.random.default_rng(self.seed).permutation(len(self))[:count]
        indices = np.asarray(list(fold_indices))
        if indices.dtype.kind not in "iu" or (indices < 0).any() or (indices >= len(self)).any():
            raise ValueError("Validation indices must be valid integer patient-row indices.")
        validation = self.metadata.index.isin(indices)
        if not validation.any() or validation.all():
            raise ValueError("Both training and validation patients are required.")
        training_x = self.x[~validation]
        mean = training_x.mean(axis=0) if scale else np.zeros(self.input_dim)
        std = training_x.std(axis=0) if scale else np.ones(self.input_dim)
        std[std < 1e-8] = 1
        scaled_x = (self.x - mean) / std
        splits = []
        for mask in (~validation, validation):
            splits.append(MutationSplitDataset(
                scaled_x[mask], self.y[mask], self.metadata.loc[mask, "patient_id"],
                self.genomic_feature_cols, self.mutation_gene_cols,
            ))
        scaler = {"mean": mean, "std": std, "feature_names": self.genomic_feature_cols}
        return splits[0], splits[1], scaler
