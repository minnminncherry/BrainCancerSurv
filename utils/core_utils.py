import torch
import torch.nn as nn
from model.model_MLPGenomic import MLPGenomics
from model.model_KmeanGenomic import KMeansGenomics
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle
import pandas as pd
from .general_utils import _get_split_loader 
from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)

def _get_result_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result"))

def _save_pkl(filename, save_object):
    with open(filename, "wb") as writer:
        pickle.dump(save_object, writer)

def _train_val(args, train_dataset, test_dataset, cur):
    # Placeholder: returns selected criterion for now.
    loss_func = _init_loss_function(args)

    model = _init_model(args)

    optimizer = _init_optim(args, model)

    train_loader, val_loader = _init_loader(args, train_dataset, test_dataset)

    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    results_dict, (total_acc, total_loss, val_acc, val_loss), best_model_path = train_model(cur, args, loss_func, model, optimizer, lr_scheduler, train_loader, val_loader)
    # save_training_history_csv(args, results_dict, cur=cur)
    eval_results, final_val_acc, final_val_loss = _evaluate_classification(
        cur,
        model,
        val_loader,
        loss_func,
        checkpoint_path=best_model_path
    )
    return results_dict, (total_acc, total_loss, final_val_acc, final_val_loss), best_model_path, eval_results

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


def compute_loss(logits, targets, criterion):
    """
    Compute loss with safe dtype shaping for common genomic setups.
    """
    if isinstance(criterion, nn.CrossEntropyLoss):
        # logits: [N, C], targets: [N]
        return criterion(logits, targets.long())
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        # logits/targets: [N] or [N, 1]
        return criterion(logits.float(), targets.float())
    return criterion(logits, targets)

def _init_model(args):
    if args.type_of_pathway == "hallmark":
        genomic_input_dim = 4371
    
    if args.modality in {'mlp', 'omics', 'snn', 'mlp_per_path'}:
        dropout = float(getattr(args, "encoder_dropout", 0.1))
        model_dict = {
             "input_dim": genomic_input_dim,
             "output_dim": int(args.n_classes),
             "projection_dim": 64,
             "dropout": dropout,
        }
        model = MLPGenomics(**model_dict)
    elif args.modality == "kmeans":
        dropout = float(getattr(args, "encoder_dropout", 0.1))
        model_dict = {
            "input_dim": genomic_input_dim,
            "output_dim": int(args.n_classes),
            "num_clusters": int(getattr(args, "num_clusters", 8)),
            "projection_dim": 64,
            "dropout": dropout,
        }
        model = KMeansGenomics(**model_dict)
    else:
        raise NotImplementedError(f"Modality {args.modality} not implemented")
    
    return model

def _init_optim(args, model):
    print("arg optimizer:" , args.opt)
    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    return optimizer

# def _init_loader(args, train_dataset, test_dataset):
#     """
#     Build simple DataLoaders from split dicts:
#     train_dataset/test_dataset = {"x": np.ndarray, "y": np.ndarray, ...}
#     """
#     batch_size = int(getattr(args, "batch_size", 32))
#     num_workers = int(getattr(args, "num_workers", 0))

#     x_train = torch.tensor(train_dataset["x"], dtype=torch.float32)
#     y_train = torch.tensor(train_dataset["y"], dtype=torch.long)
#     x_test = torch.tensor(test_dataset["x"], dtype=torch.float32)
#     y_test = torch.tensor(test_dataset["y"], dtype=torch.long)

#     train_ds = TensorDataset(x_train, y_train)
#     val_ds = TensorDataset(x_test, y_test)

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#     )
#     return train_loader, val_loader

def _init_loader(args, train_dataset, test_dataset):
    batch_size = int(getattr(args, "batch_size"))
    train_loader = _get_split_loader(args, train_dataset, training=True, batch_size=batch_size)
    val_loader = _get_split_loader(args, test_dataset, testing=True, batch_size=batch_size)
    return train_loader, val_loader

def _get_lr_scheduler(args, optimizer, dataloader):
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
    total_correct = 0
    total_count = 0
    sample_counter = 0
    patient_results = {}
    output_rows = []

    with torch.no_grad():
        for batch in val_loader:
            img_batch, x_batch, y_batch, event_time_batch, censor_batch, clinical_data_list = batch

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = compute_loss(logits, y_batch, loss_func)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            batch_size = y_batch.size(0)
            total_loss_sum += float(loss.item()) * batch_size
            total_correct += int((preds == y_batch).sum().item())
            total_count += int(batch_size)

            probs_np = probs.detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            labels_np = y_batch.detach().cpu().numpy()
            event_time_np = event_time_batch.detach().cpu().numpy()
            censor_np = censor_batch.detach().cpu().numpy()

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
                }

                row = {
                    "case_id": case_id,
                    "label": int(labels_np[batch_idx]),
                    "prediction": int(preds_np[batch_idx]),
                    "correct": bool(preds_np[batch_idx] == labels_np[batch_idx]),
                    "event_time": float(event_time_np[batch_idx]),
                    "censorship": float(censor_np[batch_idx]),
                }
                for class_idx, prob in enumerate(probs_np[batch_idx]):
                    row[f"prob_class_{class_idx}"] = float(prob)
                output_rows.append(row)
                sample_counter += 1

    final_val_loss = total_loss_sum / max(total_count, 1)
    final_val_acc = total_correct / max(total_count, 1)

    result_dir = _get_result_dir()
    checkpoint_dir = os.path.join(result_dir, "model_checkpoints")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    pkl_path = os.path.join(checkpoint_dir, f"split_{cur}_results.pkl")
    csv_path = os.path.join(result_dir, f"split_{cur}_predictions.csv")
    _save_pkl(pkl_path, patient_results)
    pd.DataFrame(output_rows).to_csv(csv_path, index=False)

    print(f"[Fold {cur}] Final evaluation saved to: {pkl_path}")
    print(f"[Fold {cur}] Prediction table saved to: {csv_path}")
    print(f"[Fold {cur}] Final val_acc={final_val_acc:.4f}, val_loss={final_val_loss:.4f}")

    return patient_results, final_val_acc, final_val_loss


def train_model(cur, args, loss_func, model, optimizer, lr_scheduler, train_loader, val_loader):
    """
    Training loop with model checkpointing.
    Returns epoch history and final epoch metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    epochs = int(getattr(args, "epoch", 1))
    best_val_loss = float('inf')
    
    # Create model save directory
    model_save_dir = os.path.join(_get_result_dir(), "model_checkpoints")
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, f"best_model_fold_{cur}.pt")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch_idx in range(epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for batch in train_loader:
            if args.modality in {"omics", "snn", "mlp_per_path", "mlp", "kmeans"}:
                _, x_batch, y_batch, _, _, _ = batch
            else:
                x_batch, y_batch = batch

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = compute_loss(logits, y_batch, loss_func)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            batch_size = y_batch.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            preds = torch.argmax(logits, dim=1)
            train_correct += int((preds == y_batch).sum().item())
            train_count += int(batch_size)

        train_loss = train_loss_sum / max(train_count, 1)
        train_acc = train_correct / max(train_count, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                if args.modality in {"omics", "snn", "mlp_per_path", "mlp", "kmeans"}:
                    _, x_batch, y_batch, _, _, _ = batch
                else:
                    x_batch, y_batch = batch

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)
                loss = compute_loss(logits, y_batch, loss_func)

                batch_size = y_batch.size(0)
                val_loss_sum += float(loss.item()) * batch_size
                preds = torch.argmax(logits, dim=1)
                val_correct += int((preds == y_batch).sum().item())
                val_count += int(batch_size)

        val_loss = val_loss_sum / max(val_count, 1)
        val_acc = val_correct / max(val_count, 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"[Fold {cur}] Epoch {epoch_idx + 1}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # Save the best checkpoint seen so far.
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), best_model_path)
        #     print(f"  -> Model saved (best val_loss: {best_val_loss:.4f})")

    total_acc = history["train_acc"][-1] if history["train_acc"] else 0.0
    total_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
    val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
    val_loss = history["val_loss"][-1] if history["val_loss"] else 0.0
    
    return history, (total_acc, total_loss, val_acc, val_loss), best_model_path


# def save_training_history_csv(args, history, cur=0, output_dir=None):
#     """
#     Save epoch-wise history and per-fold best metrics to CSV.
#     """
#     if output_dir is None:
#         output_dir = _get_result_dir()
#     os.makedirs(output_dir, exist_ok=True)

#     df = pd.DataFrame(
#         {
#             "epoch": list(range(1, len(history.get("train_loss", [])) + 1)),
#             "train_loss": history.get("train_loss", []),
#             "train_acc": history.get("train_acc", []),
#             "val_loss": history.get("val_loss", []),
#             "val_acc": history.get("val_acc", []),
#         }
#     )
#     save_path = os.path.join(output_dir, f"training_history_fold_{cur}.csv")
#     df.to_csv(save_path, index=False)
#     print(f"Saved training history to: {save_path}")

#     # Per-fold best summary (greatest accuracy, smallest loss).
#     if len(df) > 0:
#         best_train_acc_idx = int(df["train_acc"].idxmax())
#         best_val_acc_idx = int(df["val_acc"].idxmax())
#         best_train_loss_idx = int(df["train_loss"].idxmin())
#         best_val_loss_idx = int(df["val_loss"].idxmin())

#         summary_row = {
#             "fold": cur,
#             "best_train_acc": float(df.loc[best_train_acc_idx, "train_acc"]),
#             "best_train_acc_epoch": int(df.loc[best_train_acc_idx, "epoch"]),
#             "best_val_acc": float(df.loc[best_val_acc_idx, "val_acc"]),
#             "best_val_acc_epoch": int(df.loc[best_val_acc_idx, "epoch"]),
#             "best_train_loss": float(df.loc[best_train_loss_idx, "train_loss"]),
#             "best_train_loss_epoch": int(df.loc[best_train_loss_idx, "epoch"]),
#             "best_val_loss": float(df.loc[best_val_loss_idx, "val_loss"]),
#             "best_val_loss_epoch": int(df.loc[best_val_loss_idx, "epoch"]),
#         }

#         # Keep one row per fold in a shared summary file.
#         summary_path = os.path.join(output_dir, "training_best_by_fold.csv")
#         if os.path.exists(summary_path):
#             summary_df = pd.read_csv(summary_path)
#             summary_df = summary_df[summary_df["fold"] != cur]
#             summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
#         else:
#             summary_df = pd.DataFrame([summary_row])

#         summary_df = summary_df.sort_values("fold").reset_index(drop=True)
#         summary_df.to_csv(summary_path, index=False)
#         print(
#             f"[Fold {cur}] best_val_acc={summary_row['best_val_acc']:.4f} "
#             f"(epoch {summary_row['best_val_acc_epoch']}), "
#             f"best_val_loss={summary_row['best_val_loss']:.4f} "
#             f"(epoch {summary_row['best_val_loss_epoch']})"
#         )
#         print(f"Saved per-fold best summary to: {summary_path}")

#     return save_path

def save_final_fold_summary(fold_metrics, output_dir=None):
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
            "final_val_acc": float(summary_df["final_val_acc"].mean()) if len(summary_df) else 0.0,
            "final_val_loss": float(summary_df["final_val_loss"].mean()) if len(summary_df) else 0.0,
            "best_model_path": "",
        }]
    )
    summary_df = pd.concat([summary_df, average_row], ignore_index=True)

    summary_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved final cross-fold summary to: {summary_path}")
    return summary_path

# def calculate_average_accuracy_model_path(args):
#     # calculate the average accuracy across all folds and return the average accuracy of training and validation.
#     summary_path = os.path.join(os.getcwd(), "../result")
#     avg_acc = 0.0
#     for model_path in model_paths:

    
#     return avg_acc
