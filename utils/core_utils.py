import torch
import torch.nn as nn
from model.model_MLPGenomic import MLPGenomics
from model.model_SNN import SNNGenomics
from model.model_Gen2vec import Gen2VecGenomics
import os
import pickle
import pandas as pd
import torch.optim as optim
from .general_utils import _get_split_loader 
from transformers import get_linear_schedule_with_warmup

def _get_result_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result"))

def _save_pkl(filename, save_object):
    with open(filename, "wb") as writer:
        pickle.dump(save_object, writer)

def _train_val(args, train_dataset, test_dataset, cur):

    loss_func = _init_loss_function(args)

    model = _init_model(args)

    optimizer = _init_optim(args, model)

    train_loader, val_loader = _init_loader(args, train_dataset, test_dataset)

    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    results_dict, (train_cindex, total_loss), best_model_path = train_model(
        cur,
        args,
        loss_func,
        model,
        optimizer,
        lr_scheduler,
        train_loader
    )
    eval_results, final_val_cindex, final_val_loss = _evaluate_classification(
        cur,
        model,
        val_loader,
        loss_func,
        checkpoint_path=best_model_path
    )
    return results_dict, (train_cindex, total_loss, final_val_cindex, final_val_loss), best_model_path, eval_results

def _init_loss_function(args):
    """
    Simple loss selector for genomic classification.
    - cross_entropy (default): multi-class classification
    - bce: binary classification with sigmoid outputs
    """
    loss_name = args.loss_func
    loss_name = str(loss_name).strip().lower()

    if loss_name in {"cross_entropy", "ce", ""}:
        return nn.CrossEntropyLoss()
    if loss_name in {"bce", "bcewithlogits"}:
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unsupported loss_func: {loss_name}")


def prediction_probs(logits):
    return torch.softmax(logits, dim=1)


def risk_scores(logits):
    probs = torch.softmax(logits, dim=1)
    class_ids = torch.arange(logits.size(1), device=logits.device, dtype=probs.dtype)
    return -torch.sum(probs * class_ids, dim=1)


def concordance_index(event_times, censorships, risks):
    """
    Harrell-style C-index. `censorship=0` means event observed, `1` means censored.
    Higher risk should correspond to shorter survival time.
    """
    event_times = list(event_times)
    censorships = list(censorships)
    risks = list(risks)
    concordant = 0.0
    comparable = 0

    for i in range(len(event_times)):
        for j in range(i + 1, len(event_times)):
            t_i, t_j = float(event_times[i]), float(event_times[j])
            c_i, c_j = float(censorships[i]), float(censorships[j])
            r_i, r_j = float(risks[i]), float(risks[j])

            if t_i == t_j:
                continue
            if t_i < t_j and c_i == 0.0:
                comparable += 1
                concordant += 1.0 if r_i > r_j else 0.5 if r_i == r_j else 0.0
            elif t_j < t_i and c_j == 0.0:
                comparable += 1
                concordant += 1.0 if r_j > r_i else 0.5 if r_i == r_j else 0.0

    return concordant / comparable if comparable > 0 else 0.0

def _init_model(args):
    if hasattr(args, "data_factory") and hasattr(args.data_factory, "genomic_feature_cols"):
        genomic_input_dim = len(args.data_factory.genomic_feature_cols)
    elif args.type_of_pathway == "hallmark":
        genomic_input_dim = 4371
    else:
        raise ValueError("Unable to determine genomic input dimension from the dataset.")
    
    if args.modality == 'mlp':
        dropout = float(getattr(args, "encoder_dropout", 0.1))
        model_dict = {
             "input_dim": genomic_input_dim,
             "n_classes": int(args.n_classes),
             "projection_dim": 64,
             "dropout": dropout,
        }
        model = MLPGenomics(**model_dict)
    elif args.modality == "snn":
        dropout = float(getattr(args, "encoder_dropout", 0.25))
        model_dict = {
            "input_dim": genomic_input_dim,
            "n_classes": int(args.n_classes),
            "hidden_dim": 256,
            "dropout": dropout,
        }
        model = SNNGenomics(**model_dict)
    elif args.modality == "gen2vec":
        dropout = float(getattr(args, "encoder_dropout", 0.25))
        model_dict = {
            "input_dim": genomic_input_dim,
            "n_classes": int(args.n_classes),
            "embedding_dim": int(getattr(args, "gen2vec_embedding_dim", 64)),
            "hidden_dim": int(getattr(args, "gen2vec_hidden_dim", 128)),
            "dropout": dropout,
        }
        model = Gen2VecGenomics(**model_dict)
    else:
        raise NotImplementedError(f"Modality {args.modality} not implemented")
    
    return model

def _init_optim(args, model):
    if model is None:
        return None
    print("arg optimizer:" , args.opt)
    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    return optimizer

def _init_loader(args, train_dataset, test_dataset):
    if args.modality == "xgboost":
        return None, None
    batch_size = int(getattr(args, "batch_size"))
    train_loader = _get_split_loader(args, train_dataset, training=True, batch_size=batch_size)
    val_loader = _get_split_loader(args, test_dataset, testing=True, batch_size=batch_size)
    return train_loader, val_loader

def _get_lr_scheduler(args, optimizer, dataloader):
    if optimizer is None or dataloader is None:
        return None
    epoch = args.epoch
    warmup_epochs = 1
    warmup_steps = warmup_epochs * len(dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epoch,
        )

    return lr_scheduler

def _extract_case_id(clinical_data, default_id):
    if isinstance(clinical_data, dict):
        for key in ("patient_id", "_PATIENT", "sampleID", "bcr_patient_barcode"):
            if key in clinical_data and pd.notna(clinical_data[key]):
                return str(clinical_data[key])
    return f"sample_{default_id}"

def _evaluate_classification(cur, model, val_loader, loss_func, checkpoint_path=None):
    """
    Run one final evaluation pass for a fold, save per-sample predictions,
    and return the final validation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    total_loss_sum = 0.0
    total_count = 0
    all_event_times = []
    all_censorships = []
    all_risks = []
    sample_counter = 0
    patient_results = {}
    output_rows = []
    with torch.no_grad():
        for batch in val_loader:
            _, x_batch, y_batch, event_time_batch, censor_batch, clinical_data_list = batch

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            event_time_batch = event_time_batch.to(device)
            censor_batch = censor_batch.to(device)

            logits = model(x_batch)
            if isinstance(loss_func, nn.BCEWithLogitsLoss):
                loss = loss_func(logits.float(), y_batch.float())
            else:
                loss = loss_func(logits, y_batch.long())
            probs = prediction_probs(logits)
            preds = torch.argmax(probs, dim=1)
            target_bins = y_batch.long()
            risks = risk_scores(logits)

            batch_size = y_batch.size(0)
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

            for batch_idx in range(batch_size):
                case_id = _extract_case_id(clinical_data_list[batch_idx], sample_counter)
                patient_results[case_id] = {
                    "label": int(labels_np[batch_idx]),
                    "prediction": int(preds_np[batch_idx]),
                    "correct": bool(preds_np[batch_idx] == labels_np[batch_idx]),
                    "event_time": float(event_time_np[batch_idx]),
                    "censorship": float(censor_np[batch_idx]),
                    "clinical": clinical_data_list[batch_idx],
                    "logits": logits_np[batch_idx],
                    "probabilities": probs_np[batch_idx],
                    "risk": float(risks_np[batch_idx]),
                }
                row = {
                    "case_id": case_id,
                    "label": int(labels_np[batch_idx]),
                    "prediction": int(preds_np[batch_idx]),
                    "correct": bool(preds_np[batch_idx] == labels_np[batch_idx]),
                    "event_time": float(event_time_np[batch_idx]),
                    "censorship": float(censor_np[batch_idx]),
                    "risk": float(risks_np[batch_idx]),
                }
                for class_idx, prob in enumerate(probs_np[batch_idx]):
                    row[f"prob_class_{class_idx}"] = float(prob)
                output_rows.append(row)

                sample_counter += 1

    final_val_loss = total_loss_sum / max(total_count, 1)
    final_val_cindex = concordance_index(all_event_times, all_censorships, all_risks)

    result_dir = _get_result_dir()
    checkpoint_dir = os.path.join(result_dir, "model_checkpoints")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    pkl_path = os.path.join(checkpoint_dir, f"split_{cur}_results.pkl")
    csv_path = os.path.join(result_dir, f"split_{cur}_predictions.csv")
    pd.DataFrame(output_rows).to_csv(csv_path, index=False)
    _save_pkl(pkl_path, patient_results)


    print(f"[Fold {cur}] Final evaluation saved to: {pkl_path}")
    print(f"[Fold {cur}] Final val_cindex={final_val_cindex:.4f}, val_loss={final_val_loss:.4f}")

    return patient_results, final_val_cindex, final_val_loss


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

    history = {
        "train_loss": [],
        "train_cindex": [],
    }

    for epoch_idx in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        all_event_times = []
        all_censorships = []
        all_risks = []

        for batch in train_loader:
            if args.modality in {"mlp", "omics", "mlp_per_path", "snn", "gen2vec", "kmeans"}:
                _, x_batch, y_batch, event_time_batch, censor_batch, _ = batch

            # print(f"Batch x shape: {x_batch.shape}, y shape: {y_batch.shape}")
            # print(f"Header of batch x: {x_batch[0][:5]}, batch y: {y_batch[0]}")

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if event_time_batch is not None:
                event_time_batch = event_time_batch.to(device)
            if censor_batch is not None:
                censor_batch = censor_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_func(logits, y_batch.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            batch_size = y_batch.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            probs = prediction_probs(logits)
            preds = torch.argmax(probs, dim=1)
            target_bins = y_batch.long()
            risks = risk_scores(logits)
            if event_time_batch is not None and censor_batch is not None:
                all_event_times.extend(event_time_batch.detach().cpu().numpy().tolist())
                all_censorships.extend(censor_batch.detach().cpu().numpy().tolist())
                all_risks.extend(risks.detach().cpu().numpy().tolist())
            train_count += int(batch_size)
            print(f"Batch {train_count} | Loss: {loss.item():.4f} | Batch Bin Acc: {(preds == target_bins).float().mean().item():.4f}")

        train_loss = train_loss_sum / max(train_count, 1)
        train_cindex = concordance_index(all_event_times, all_censorships, all_risks)

        history["train_loss"].append(train_loss)
        history["train_cindex"].append(train_cindex)
        print(
            f"[Fold {cur}] Epoch {epoch_idx + 1}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_cindex={train_cindex:.4f}"
        )

    torch.save(model.state_dict(), final_model_path)
    print(f"[Fold {cur}] Final model saved to: {final_model_path}")

    total_cindex = history["train_cindex"][-1] if history["train_cindex"] else 0.0
    total_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
    
    return history, (total_cindex, total_loss), final_model_path

def save_final_fold_summary(fold_metrics, model, genomic_file_name, output_dir=None):
    """
    Save one row per fold plus an average row for the final evaluation results.
    """
    if output_dir is None:
        output_dir = _get_result_dir()
    os.makedirs(output_dir, exist_ok=True)

    summary_df = pd.DataFrame(fold_metrics)
    average_row = pd.DataFrame(
        [{
            "fold": "average",
            "train_cindex": float(summary_df["train_cindex"].mean()) if len(summary_df) else 0.0,
            "train_loss": float(summary_df["train_loss"].mean()) if len(summary_df) else 0.0,
            "final_val_cindex": float(summary_df["final_val_cindex"].mean()) if len(summary_df) else 0.0,
            "final_val_loss": float(summary_df["final_val_loss"].mean()) if len(summary_df) else 0.0,
            "best_model_path": "",
        }]
    )
    summary_df = pd.concat([summary_df, average_row], ignore_index=True)

    summary_dir = os.path.join(output_dir, model, genomic_file_name)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved final cross-fold summary to: {summary_path}")
    return summary_path
