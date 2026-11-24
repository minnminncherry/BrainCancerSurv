import torch
import numpy as np
import torch.nn as nn
from model.gene_MLP_model import MLPGenomics
from model.gene_SNN_model import SNNGenomics
from model.gene_Gen2vec_model import Gen2VecGenomics
from model.gene_resnet_mlp_model import ResMLPGenomics
from model.mri_3DCNN_model import CNN3DMRI
from model.wsi_MIL_model import MILWSI
from model.wsi_transmil_model import TRANSMILWSI
from model.multimodal_late_fusion_wsi_gene import MultimodalLateFusion
from model.multimodal_cross_attn_wsi_gene import MultimodalCrossAttention
from model.mri_ViT_model import ViTMRI
from model.multimodal_mutation_survival import MutationGuidedSurvival
from model.multimodal_late_fusion_wsi_mri_gene import MultimodalWSIMRILateFusion
from model.multimodal_cross_attn_wsi_mri_gene import MultimodalWSIMRICrossAttention
from model.multimodal_late_fusion_mri_gene import MultimodalMRILateFusion
from model.multimodal_cross_attn_mri_gene import MultimodalMRICrossAttention
import os
import time
import pickle
import pandas as pd
import torch.optim as optim
from utils.loss_utils import LossClass, survival_outputs
from dataset.SurvivalMutationDataset import _prepare_mutation_labels
from utils.metrics_utils import concordance_index
from utils.tensorboard_utils import start_tensorboard, log_tensorboard
from .general_utils import _get_split_loader
from transformers import get_linear_schedule_with_warmup
from torchvision import models, transforms


def _get_result_dir():
    return os.path.abspath(os.environ.get("RESULT_DIR") or os.path.join(os.path.dirname(__file__), "..", "result"))

def _save_pkl(filename, save_object):
    with open(filename, "wb") as writer:
        pickle.dump(save_object, writer)

def _train_val(args, train_dataset, test_dataset, cur):

    # Patient splits already exist. Attach targets/masks and establish gene order
    # before _init_model uses that order to size the optional mutation head.
    _prepare_mutation_labels(args, train_dataset, test_dataset, cur)
    loss_func = _init_loss_function(args)
    if isinstance(loss_func, LossClass):
        interval_dir = os.path.join(_get_result_dir(), args.modality, args.genomic_file_name)
        os.makedirs(interval_dir, exist_ok=True)
        np.save(os.path.join(interval_dir, f"survival_bin_edges_fold_{cur}.npy"), args.data_factory.survival_bin_edges)

    model = _init_model(args)

    optimizer = _init_optim(args, model)

    train_loader, val_loader = _init_loader(args, train_dataset, test_dataset)

    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    results_dict, (train_cindex, total_loss, training_time_seconds, training_time_per_sample_seconds), best_model_path = train_model(
        cur,
        args,
        loss_func,
        model,
        optimizer,
        lr_scheduler,
        train_loader,
    )
    if isinstance(model, ResMLPGenomics):
        from utils.genomic_external import save_genomic_bundle
        bundle_dir = os.path.join(_get_result_dir(), args.modality, args.genomic_file_name)
        print(f"Saved external-test bundle: {save_genomic_bundle(args, model, train_dataset, cur, bundle_dir)}")
    eval_results, final_val_cindex, final_val_loss, inference_summary = _evaluate_classification(
        args,
        cur,
        model,
        val_loader,
        loss_func,
        checkpoint_path=best_model_path
    )
    return (
        results_dict,
        (
            train_cindex,
            total_loss,
            final_val_cindex,
            final_val_loss,
            training_time_seconds,
            training_time_per_sample_seconds,
            inference_summary["inference_time_seconds"],
            inference_summary["inference_time_per_sample_seconds"],
        ),
        best_model_path,
        eval_results,
    )

def _init_loss_function(args):
    """Select classification loss or censor-aware discrete survival NLL."""
    loss_name = args.loss_func
    loss_name = str(loss_name).strip().lower()

    if loss_name in {"cross_entropy", "ce", ""}:
        return nn.CrossEntropyLoss()
    if loss_name in {"sigmoid", "bce", "bcewithlogits"}:
        return nn.BCEWithLogitsLoss()
    if loss_name in {"nllsurvloss", "nll_surv", "nll_survival", "nll_loss", "lossclass"}:
        return LossClass()
    raise ValueError(f"Unsupported loss_func: {loss_name}")

def _classification_loss(loss_func, logits, labels, censorship=None):
    if isinstance(loss_func, LossClass):
        if censorship is None:
            raise ValueError("LossClass requires censorship labels.")
        return loss_func(logits, labels, censorship)
    if isinstance(loss_func, nn.BCEWithLogitsLoss):
        # Class indices become one-hot targets matching the model's output shape.
        targets = torch.nn.functional.one_hot(labels.long(), num_classes=logits.size(1))
        return loss_func(logits, targets.to(dtype=logits.dtype))
    return loss_func(logits, labels.long())


def _prediction_loss(args, loss_func, output, labels, censoring, clinical):
    """Keep legacy models working; add masked mutation supervision for the diagram model."""
    if not isinstance(output, dict):
        return output, _classification_loss(loss_func, output, labels, censoring)
    logits = output["survival_logits"]
    survival_loss = _classification_loss(loss_func, logits, labels, censoring)
    if not all("mutation_targets" in row and "mutation_mask" in row for row in clinical):
        raise ValueError("Mutation-guided training requires mutation labels and availability masks.")
    targets = logits.new_tensor([row["mutation_targets"] for row in clinical])
    mask = logits.new_tensor([row["mutation_mask"] for row in clinical])
    total = survival_loss
    for branch in ("genomic", "mri", "wsi"):
        if f"{branch}_mutation_logits" not in output:
            continue
        mutation_logits = output[f"{branch}_mutation_logits"]
        if mutation_logits.shape != targets.shape or mask.shape != targets.shape:
            raise ValueError("Mutation logits, target genes, and masks must have identical shapes.")
        error = nn.functional.binary_cross_entropy_with_logits(mutation_logits, targets, reduction="none")
        mutation_loss = (error * mask).sum() / mask.sum().clamp_min(1)
        total = total + getattr(args, f"{branch}_mutation_weight", 1.0) * mutation_loss
    return logits, total


def prediction_probs(logits, loss_func=None):
    if isinstance(loss_func, LossClass):
        # Interval event masses; the remaining probability is survival past the grid.
        return survival_outputs(logits)[2]
    return torch.softmax(logits, dim=1)


def risk_scores(logits, loss_func=None):
    if isinstance(loss_func, LossClass):
        return -survival_outputs(logits)[1].sum(dim=1)
    probs = torch.softmax(logits, dim=1)
    class_ids = torch.arange(logits.size(1), device=logits.device, dtype=probs.dtype)
    return -torch.sum(probs * class_ids, dim=1)

def _sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def calculate_inference_time_per_sample(model, device, sample_count, *inputs):
    """
    Run one inference forward pass and return:
    logits, total batch inference time, inference time per sample.
    """
    sample_count = max(int(sample_count), 1)
    _sync_cuda(device)
    start_time = time.perf_counter()
    logits = model(*inputs)
    _sync_cuda(device)
    total_time = time.perf_counter() - start_time
    return logits, total_time, total_time / sample_count


def calculate_training_time(start_time, device, sample_count=1):
    """
    Calculate elapsed training time from `start_time`.
    Returns total elapsed seconds and elapsed seconds per sample.
    """
    sample_count = max(int(sample_count), 1)
    _sync_cuda(device)
    total_time = time.perf_counter() - start_time
    return total_time, total_time / sample_count


def _parse_batch(batch):
    """Parse genomic, imaging, two-modality, or three-modality batches."""
    if len(batch) == 5:
        input_data, y_batch, event_time_batch, censor_batch, clinical_data_list = batch
        return input_data, None, y_batch, event_time_batch, censor_batch, clinical_data_list
    if len(batch) == 6:
        wsi_batch, omics_batch, y_batch, event_time_batch, censor_batch, clinical_data_list = batch
        return omics_batch, wsi_batch, y_batch, event_time_batch, censor_batch, clinical_data_list
    if len(batch) == 7:
        mri, wsi, genes, labels, times, censoring, clinical = batch
        return genes, (wsi, mri), labels, times, censoring, clinical
    raise ValueError(f"Unsupported batch format with {len(batch)} elements.")


def _aux_to_device(auxiliary, device):
    if auxiliary is None:
        return None
    if isinstance(auxiliary, tuple):
        return tuple(item.to(device) for item in auxiliary)
    return auxiliary.to(device)


def _model_inputs(inputs, auxiliary):
    if auxiliary is None:
        return (inputs,)
    return (inputs, *auxiliary) if isinstance(auxiliary, tuple) else (inputs, auxiliary)


def _prepare_batch(batch, device):
    """Move any supported batch to the device and return positional model inputs."""
    inputs, auxiliary, labels, times, censoring, clinical = _parse_batch(batch)
    model_inputs = _model_inputs(inputs.to(device), _aux_to_device(auxiliary, device))
    labels = labels.to(device)
    times = times.to(device) if times is not None else None
    censoring = censoring.to(device) if censoring is not None else None
    return model_inputs, labels, times, censoring, clinical


def _init_model(args):
    """Build the selected encoder/fusion model with the existing defaults."""
    if hasattr(args, "data_factory") and hasattr(args.data_factory, "genomic_feature_cols"):
        genomic_dim = len(args.data_factory.genomic_feature_cols)
    elif args.type_of_pathway == "hallmark":
        genomic_dim = 4371
    else:
        raise ValueError("Unable to determine genomic input dimension from the dataset.")

    def dropout(default):
        value = getattr(args, "encoder_dropout", None)
        return float(default if value is None else value)

    classes = int(args.n_classes)
    if args.modality in {"mutation_guided_late_fusion", "mutation_guided_cross_attention"}:
        targets = getattr(args.data_factory, "mutation_gene_cols", [])
        if not targets:
            raise ValueError("Provide --mutation_file and select mutation target genes.")
        weights = [args.mri_mutation_weight, args.wsi_mutation_weight]
        if not np.isfinite(weights).all() or min(weights) < 0:
            raise ValueError("Mutation loss weights must be finite and nonnegative.")
        return MutationGuidedSurvival(
            genomic_input_dim=genomic_dim, n_targets=len(targets), n_classes=classes,
            wsi_input_dim=args.wsi_feature_dim, embedding_dim=args.embedding_dim,
            genomic_hidden_dim=args.resmlp_hidden_dim, num_blocks=args.resmlp_num_blocks,
            dropout=dropout(0.3), fusion=args.modality.removeprefix("mutation_guided_"),
            num_heads=args.attention_heads, mri_hidden_dim=args.mri_hidden_dim,
            mri_num_layers=args.mri_num_layers, mri_patch_size=args.mri_patch_size,
        )

    if args.modality == "mlp":
        return MLPGenomics(input_dim=genomic_dim, n_classes=classes, projection_dim=4028,
                           dropout=dropout(0.2), num_layers=10)
    if args.modality == "snn":
        return SNNGenomics(input_dim=genomic_dim, n_classes=classes, hidden_dim=2048,
                           dropout=dropout(0.2), num_layers=8)
    if args.modality == "gen2vec":
        return Gen2VecGenomics(
            input_dim=genomic_dim, n_classes=classes, dropout=dropout(0.2),
            embedding_dim=int(getattr(args, "gen2vec_embedding_dim", 256)),
            hidden_dim=int(getattr(args, "gen2vec_hidden_dim", 1024)), use_zero_mask=True,
        )
    if args.modality == "resmlp":
        return ResMLPGenomics(
            input_dim=genomic_dim, n_classes=classes, dropout=dropout(0.2),
            hidden_dim=int(getattr(args, "resmlp_hidden_dim", 256)),
            num_blocks=int(getattr(args, "resmlp_num_blocks", 5)),
            n_mutation_targets=len(getattr(args.data_factory, "mutation_gene_cols", []))
            if getattr(args, "mutation_file", None) else 0,
        )
    if args.modality in {"mil", "transmil"}:
        is_mil = args.modality == "mil"
        model_class = MILWSI if is_mil else TRANSMILWSI
        return model_class(input_dim=int(getattr(args, "wsi_feature_dim", 1024 if is_mil else 512)),
                           n_classes=classes, hidden_dim=256, dropout=dropout(0.2 if is_mil else 0.5))
    if args.modality == "ViT":
        return ViTMRI(input_dim=int(getattr(args, "mri_feature_dim", 1024)),
                      n_classes=classes, hidden_dim=128, dropout=dropout(0.3), num_layers=2)
    if args.modality == "3dcnn":
        return CNN3DMRI(n_classes=classes, hidden_dim=128, dropout=dropout(0.3))

    # Model class, default hidden size, block count, dropout, and whether WSI is used.
    fusion_models = {
        "multimodal_late_fusion": (MultimodalLateFusion, 256, 5, 0.2, True),
        "multimodal_cross_attention": (MultimodalCrossAttention, 256, 5, 0.2, True),
        "multimodal_mri_late_fusion": (MultimodalMRILateFusion, 128, 2, 0.3, False),
        "multimodal_mri_cross_attention": (MultimodalMRICrossAttention, 128, 2, 0.3, False),
        "multimodal_wsi_mri_late_fusion": (MultimodalWSIMRILateFusion, 32, 1, 0.3, True),
        "multimodal_wsi_mri_cross_attention": (MultimodalWSIMRICrossAttention, 32, 1, 0.3, True),
    }
    if args.modality not in fusion_models:
        raise NotImplementedError(f"Modality {args.modality} not implemented")
    model_class, hidden, blocks, default_dropout, has_wsi = fusion_models[args.modality]
    options = dict(genomic_input_dim=genomic_dim, n_classes=classes,
                   hidden_dim=int(getattr(args, "resmlp_hidden_dim", hidden)),
                   num_blocks=int(getattr(args, "resmlp_num_blocks", blocks)),
                   dropout=dropout(default_dropout))
    if has_wsi:
        options["wsi_input_dim"] = int(getattr(args, "wsi_feature_dim", 1024))
    return model_class(**options)

def _init_optim(args, model):
    if model is None:
        return None
    print("arg optimizer:", args.opt)
    options = {"lr": args.lr, "weight_decay": getattr(args, "reg", 0.0)}
    if args.opt == "adam":
        return optim.Adam(model.parameters(), **options)
    if args.opt == "sgd":
        return optim.SGD(model.parameters(), momentum=0.9, **options)
    raise ValueError(f"Unsupported optimizer: {args.opt}")

def _init_loader(args, train_dataset, test_dataset):
    batch_size = int(getattr(args, "batch_size"))
    train_loader = _get_split_loader(args, train_dataset, training=True, batch_size=batch_size)
    val_loader = _get_split_loader(args, test_dataset, testing=True, batch_size=batch_size)
    return train_loader, val_loader

def _get_lr_scheduler(args, optimizer, dataloader):
    if optimizer is None or dataloader is None:
        return None
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(dataloader),
        num_training_steps=len(dataloader) * args.epoch,
    )

def _extract_case_id(clinical_data, default_id):
    if isinstance(clinical_data, dict):
        for key in ("patient_id", "_PATIENT", "bcr_patient_barcode"):
            if key in clinical_data and pd.notna(clinical_data[key]):
                return str(clinical_data[key])
    return f"sample_{default_id}"

def _evaluate_classification(args,cur, model, val_loader, loss_func, checkpoint_path=None):
    """
    Run one final evaluation pass for a fold, save per-sample predictions,
    and return the final validation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    total_loss_sum = 0.0
    total_count = 0
    all_event_times = []
    all_censorships = []
    all_risks = []
    sample_counter = 0
    inference_time_seconds = 0.0
    inference_sample_count = 0
    patient_results = {}
    output_rows = []
    mutation_suffix = "without_mutation"
    with torch.no_grad():
        for batch in val_loader:
            inputs, y_batch, event_time_batch, censor_batch, clinical_data_list = _prepare_batch(batch, device)
            batch_size = y_batch.size(0)
            output, batch_inference_time, batch_inference_time_per_sample = calculate_inference_time_per_sample(
                model, device, batch_size, *inputs,
            )
            inference_time_seconds += float(batch_inference_time)
            inference_sample_count += int(batch_size)
            logits, loss = _prediction_loss(args, loss_func, output, y_batch, censor_batch, clinical_data_list)
            probs = prediction_probs(logits, loss_func)
            preds = torch.argmax(probs, dim=1)
            target_bins = y_batch.long()
            risks = risk_scores(logits, loss_func)

            total_loss_sum += float(loss.item()) * batch_size
            total_count += int(batch_size)

            probs_np = probs.detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            labels_np = target_bins.detach().cpu().numpy()
            event_time_np = event_time_batch.detach().cpu().numpy()
            censor_np = censor_batch.detach().cpu().numpy()
            risks_np = risks.detach().cpu().numpy()
            all_event_times.extend(event_time_np.tolist())
            all_censorships.extend(censor_np.tolist())
            all_risks.extend(risks_np.tolist())

            mutation_suffix = "mutation" if isinstance(output, dict) and any(
                key.endswith("_mutation_logits") for key in output
            ) else "without_mutation"
            for batch_idx in range(batch_size):
                case_id = _extract_case_id(clinical_data_list[batch_idx], sample_counter)
                patient_results[case_id] = {
                    "label": int(labels_np[batch_idx]),
                    "prediction": int(preds_np[batch_idx]),
                    "correct": bool(preds_np[batch_idx] == labels_np[batch_idx]),
                    "event_time": float(event_time_np[batch_idx]),
                    "censorship": float(censor_np[batch_idx]),
                    "logits": logits_np[batch_idx],
                    "probabilities": probs_np[batch_idx],
                    "risk": float(risks_np[batch_idx]),
                    "inference_time_seconds": float(batch_inference_time_per_sample),
                }
                row = {
                    "case_id": case_id,
                    "label": int(labels_np[batch_idx]),
                    "prediction": int(preds_np[batch_idx]),
                    "correct": bool(preds_np[batch_idx] == labels_np[batch_idx]),
                    "event_time": float(event_time_np[batch_idx]),
                    "censorship": float(censor_np[batch_idx]),
                    "risk": float(risks_np[batch_idx]),
                    "inference_time_seconds": float(batch_inference_time_per_sample),
                }
                for class_idx, prob in enumerate(probs_np[batch_idx]):
                    row[f"prob_class_{class_idx}"] = float(prob)
                if isinstance(loss_func, LossClass):
                    hazards, survival, _ = survival_outputs(logits[batch_idx:batch_idx + 1])
                    row["prediction_type"] = "discrete_survival"
                    row["event_bin_observed"] = bool(censor_np[batch_idx] == 0)
                    patient_results[case_id]["event_bin_observed"] = row["event_bin_observed"]
                    row["survival_tail_probability"] = float(survival[0, -1].item())
                    for interval in range(logits.size(1)):
                        row[f"hazard_{interval}"] = float(hazards[0, interval].item())
                        row[f"survival_{interval}"] = float(survival[0, interval].item())
                    patient_results[case_id]["hazards"] = hazards[0].cpu().numpy()
                    patient_results[case_id]["survival"] = survival[0].cpu().numpy()
                clinical = clinical_data_list[batch_idx]
                if "mutation_targets" in clinical:
                    for gene, target, known in zip(args.data_factory.mutation_gene_cols,
                                                   clinical["mutation_targets"], clinical["mutation_mask"]):
                        row[f"mutation_label_{gene}"] = int(target) if known else None
                        row[f"mutation_known_{gene}"] = bool(known)
                if isinstance(output, dict):
                    for branch in ("genomic", "mri", "wsi"):
                        if f"{branch}_mutation_logits" not in output:
                            continue
                        mutation_probs = output[f"{branch}_mutation_logits"][batch_idx].sigmoid().cpu().numpy()
                        if f"{branch}_embedding" in output:
                            patient_results[case_id][f"{branch}_embedding"] = output[f"{branch}_embedding"][batch_idx].cpu().numpy()
                        patient_results[case_id][f"{branch}_mutation_probabilities"] = mutation_probs
                        for gene, probability in zip(args.data_factory.mutation_gene_cols, mutation_probs):
                            row[f"{branch}_mutation_prob_{gene}"] = float(probability)
                output_rows.append(row)

                sample_counter += 1

    final_val_loss = total_loss_sum / max(total_count, 1)
    final_val_cindex = concordance_index(all_event_times, all_censorships, all_risks)
    observed_rows = [row for row in output_rows if row["censorship"] == 0]
    log_tensorboard(args, getattr(args, "epoch", 0), {
        "C-index/validation_final": final_val_cindex,
        "Loss/validation_total_final": final_val_loss,
        "Accuracy/validation_observed_bin_final":
            np.mean([row["correct"] for row in observed_rows]) if observed_rows else None,
    })
    inference_summary = {
        "inference_time_seconds": float(inference_time_seconds),
        "inference_time_per_sample_seconds": float(inference_time_seconds / max(inference_sample_count, 1)),
        "num_inference_samples": int(inference_sample_count),
    }

    result_dir = _get_result_dir()
    checkpoint_dir = os.path.join(result_dir, "model_checkpoints")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    pkl_path = os.path.join(checkpoint_dir, f"split_{args.modality}_{args.genomic_file_name}_{cur}_results.pkl")
    csv_path = os.path.join(result_dir, f"split_{args.modality}_{args.genomic_file_name}_{cur}_{mutation_suffix}_predictions.csv")
    pd.DataFrame(output_rows).to_csv(csv_path, index=False)
    _save_pkl(pkl_path, patient_results)


    print(f"[Fold {cur}] Final evaluation saved to: {pkl_path}")
    print(f"[Fold {cur}] Final val_cindex={final_val_cindex:.4f}, val_loss={final_val_loss:.4f}")
    print(
        f"[Fold {cur}] Inference time={inference_summary['inference_time_seconds']:.4f}s | "
        f"per_sample={inference_summary['inference_time_per_sample_seconds']:.6f}s"
    )

    return patient_results, final_val_cindex, final_val_loss, inference_summary


def train_model(cur, args, loss_func, model, optimizer, lr_scheduler, train_loader):
    """
    Simple training loop. Evaluation is handled separately after training.
    Returns training history and the final checkpoint path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    epochs = int(getattr(args, "epoch", 1))

    model_save_dir = os.path.join(_get_result_dir(), "model_checkpoints")
    os.makedirs(model_save_dir, exist_ok=True)
    final_model_path = os.path.join(model_save_dir, f"best_model_fold_{cur}.pt")
    history_dir = os.path.join(_get_result_dir(), args.modality, args.genomic_file_name)
    os.makedirs(history_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "train_cindex": [],
        "loss_func": [],
        "train_survival_loss": [],
        "train_mutation_loss": [],
    }
    if isinstance(model, MutationGuidedSurvival):
        history["mri_mutation_weight"] = []
        history["wsi_mutation_weight"] = []
        history["embedding_dim"] = []
    training_start_time = time.perf_counter()
    training_sample_count = 0
    mutation_mode = "mutation" if isinstance(model, MutationGuidedSurvival) or (
        isinstance(model, ResMLPGenomics) and model.mutation_head is not None
    ) else "without_mutation"
    start_tensorboard(args, cur, mutation_mode)

    for epoch_idx in range(epochs):
        model.train()
        train_loss_sum = 0.0
        survival_loss_sum = 0.0
        train_count = 0
        observed_correct, observed_count = 0, 0
        all_event_times = []
        all_censorships = []
        all_risks = []

        for batch in train_loader:
            inputs, y_batch, event_time_batch, censor_batch, clinical = _prepare_batch(batch, device)
            optimizer.zero_grad()
            logits, loss = _prediction_loss(args, loss_func, model(*inputs), y_batch, censor_batch, clinical)
            with torch.no_grad():
                survival_loss = _classification_loss(loss_func, logits.detach(), y_batch, censor_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            batch_size = y_batch.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            survival_loss_sum += float(survival_loss.item()) * batch_size
            probs = prediction_probs(logits, loss_func)
            preds = torch.argmax(probs, dim=1)
            target_bins = y_batch.long()
            observed = censor_batch == 0
            observed_correct += int(((preds == target_bins) & observed).sum().item())
            observed_count += int(observed.sum().item())
            risks = risk_scores(logits, loss_func)
            if event_time_batch is not None and censor_batch is not None:
                all_event_times.extend(event_time_batch.detach().cpu().numpy().tolist())
                all_censorships.extend(censor_batch.detach().cpu().numpy().tolist())
                all_risks.extend(risks.detach().cpu().numpy().tolist())
            train_count += int(batch_size)
            training_sample_count += int(batch_size)
            print(f"Batch {train_count} | Loss: {loss.item():.4f} | Batch Bin Acc: {(preds == target_bins).float().mean().item():.4f}")

        train_loss = train_loss_sum / max(train_count, 1)
        train_cindex = concordance_index(all_event_times, all_censorships, all_risks)

        history["train_loss"].append(train_loss)
        history["train_survival_loss"].append(survival_loss_sum / max(train_count, 1))
        history["train_mutation_loss"].append(train_loss - history["train_survival_loss"][-1])
        history["train_cindex"].append(train_cindex)
        history["loss_func"].append(type(loss_func).__name__)
        log_tensorboard(args, epoch_idx + 1, {
            "C-index/train": train_cindex,
            "Loss/train_total": train_loss,
            "Loss/train_survival": history["train_survival_loss"][-1],
            "Loss/train_mutation_weighted": history["train_mutation_loss"][-1],
            "Accuracy/train_observed_bin": observed_correct / observed_count if observed_count else None,
            "Optimizer/learning_rate": optimizer.param_groups[0]["lr"],
        })
        if isinstance(model, MutationGuidedSurvival):
            history["mri_mutation_weight"].append(args.mri_mutation_weight)
            history["wsi_mutation_weight"].append(args.wsi_mutation_weight)
            history["embedding_dim"].append(model.embedding_dim)
        print(
            f"[Fold {cur}] Epoch {epoch_idx + 1}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_cindex={train_cindex:.4f}"
        )

        pd.DataFrame(history).to_csv(os.path.join(history_dir, f"training_history_fold_{cur}.csv"), index=False)
        pd.DataFrame(history).to_csv(os.path.join(history_dir, f"training_history_fold_{cur}_{mutation_mode}.csv"), index=False)

    torch.save(model.state_dict(), final_model_path)
    history["selected_epoch"] = [len(history["train_loss"])] * len(history["train_loss"])
    pd.DataFrame(history).to_csv(os.path.join(history_dir, f"training_history_fold_{cur}.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(history_dir, f"training_history_fold_{cur}_{mutation_mode}.csv"), index=False)
    print(f"[Fold {cur}] Selected model saved to: {final_model_path}")

    total_cindex = history["train_cindex"][-1] if history["train_cindex"] else 0.0
    total_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
    training_time_seconds, training_time_per_sample_seconds = calculate_training_time(
        training_start_time,
        device,
        training_sample_count,
    )
    print(
        f"[Fold {cur}] Training time={training_time_seconds:.4f}s | "
        f"per_sample={training_time_per_sample_seconds:.6f}s"
    )

    return history, (total_cindex, total_loss, training_time_seconds, training_time_per_sample_seconds), final_model_path


def save_final_fold_summary(fold_metrics, model, genomic_file_name, output_dir=None):
    """
    Save one row per fold plus an average row for the final evaluation results.
    """
    if output_dir is None:
        output_dir = _get_result_dir()
    os.makedirs(output_dir, exist_ok=True)

    summary_df = pd.DataFrame(fold_metrics)
    average_values = {
        "fold": "average",
        "train_cindex": float(summary_df["train_cindex"].mean()) if len(summary_df) else 0.0,
        "train_loss": float(summary_df["train_loss"].mean()) if len(summary_df) else 0.0,
        "final_val_cindex": float(summary_df["final_val_cindex"].mean()) if len(summary_df) else 0.0,
        "final_val_loss": float(summary_df["final_val_loss"].mean()) if len(summary_df) else 0.0,
        "best_model_path": "",
    }
    for col in (
        "training_time_seconds",
        "training_time_per_sample_seconds",
        "inference_time_seconds",
        "inference_time_per_sample_seconds",
    ):
        if col in summary_df.columns:
            average_values[col] = float(summary_df[col].mean()) if len(summary_df) else 0.0

    average_row = pd.DataFrame([average_values])
    summary_df = pd.concat([summary_df, average_row], ignore_index=True)

    summary_dir = os.path.join(output_dir, model, genomic_file_name)
    os.makedirs(summary_dir, exist_ok=True)
    # Read the objective recorded during each fold, rather than guessing from its loss value.
    losses = []
    for fold in summary_df["fold"]:
        if fold == "average":
            losses.append(losses[0] if losses and len(set(losses)) == 1 else "Mixed/Unknown")
            continue
        history_path = os.path.join(summary_dir, f"training_history_fold_{int(fold)}.csv")
        history = pd.read_csv(history_path) if os.path.isfile(history_path) else pd.DataFrame()
        names = history["loss_func"].dropna().unique() if "loss_func" in history else []
        losses.append(names[0] if len(names) == 1 else "Unknown")
    summary_df["loss_func"] = losses
    summary_path = os.path.join(summary_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved final cross-fold summary to: {summary_path}")
    return summary_path

# encoder for WSI patch feature extraction, e.g. resnet50 or conch
def encoder(model_name, target_img_size=224):

    model_name = str(model_name).lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        feature_encoder = models.resnet50(weights=weights).to(device)
        preprocessing = weights.transforms()
        mean, std = preprocessing.mean, preprocessing.std
        feature_encoder.fc = nn.Identity()

    elif model_name == "conch":
        import timm

        feature_encoder = timm.create_model("conch_base", pretrained=True)
        config = timm.data.resolve_data_config({}, model=feature_encoder)
        mean, std = config["mean"], config["std"]
        feature_encoder = feature_encoder.to(device)
        if hasattr(feature_encoder, "head"):
            feature_encoder.head = nn.Identity()
        elif hasattr(feature_encoder, "fc"):
            feature_encoder.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    for param in feature_encoder.parameters():
        param.requires_grad = False
    feature_encoder.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((target_img_size, target_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return feature_encoder, img_transforms
