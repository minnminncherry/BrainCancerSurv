import torch
import torch.nn as nn
from model.model_MLPGenomic import MLPGenomics
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd
from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)

def _train_val(args, train_dataset, test_dataset, cur):
    # Placeholder: returns selected criterion for now.
    loss_func = _init_loss_function(args)

    model = _init_model(args)

    optimizer = _init_optim(args, model)

    train_loader, val_loader = _init_loader(args, train_dataset, test_dataset)

    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    results_dict, (total_acc, total_loss, val_acc, val_loss) = train_model(cur, args, loss_func, model, optimizer, lr_scheduler, train_loader, val_loader)
    save_training_history_csv(args, results_dict, cur=cur)
    return results_dict, (total_acc, total_loss, val_acc, val_loss)

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
    
    if args.modality == 'mlp':
        dropout = float(getattr(args, "encoder_dropout", 0.1))
        model_dict = {
             "input_dim": genomic_input_dim,
             "output_dim": int(args.n_classes),
             "projection_dim": 64,
             "dropout": dropout,
        }
        model = MLPGenomics(**model_dict)
    
    return model

def _init_optim(args, model):
    print("arg optimizer:" , args.opt)
    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001) #change with variable
    
    return optimizer

def _init_loader(args, train_dataset, test_dataset):
    """
    Build simple DataLoaders from split dicts:
    train_dataset/test_dataset = {"x": np.ndarray, "y": np.ndarray, ...}
    """
    batch_size = int(getattr(args, "batch_size", 32))
    num_workers = int(getattr(args, "num_workers", 0))

    x_train = torch.tensor(train_dataset["x"], dtype=torch.float32)
    y_train = torch.tensor(train_dataset["y"], dtype=torch.long)
    x_test = torch.tensor(test_dataset["x"], dtype=torch.float32)
    y_test = torch.tensor(test_dataset["y"], dtype=torch.long)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
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


def train_model(cur, args, loss_func, model, optimizer, lr_scheduler, train_loader, val_loader):
    """
    Simple training loop using epoch count from args.epoch.
    Returns epoch history and final epoch metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    epochs = int(getattr(args, "epoch", 1))

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

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = compute_loss(logits, y_batch, loss_func)
            loss.backward()
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
            for x_batch, y_batch in val_loader:
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

    total_acc = history["train_acc"][-1] if history["train_acc"] else 0.0
    total_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
    val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
    val_loss = history["val_loss"][-1] if history["val_loss"] else 0.0
    return history, (total_acc, total_loss, val_acc, val_loss)


def save_training_history_csv(args, history, cur=0, output_dir=None):
    """
    Save epoch-wise history and per-fold best metrics to CSV.
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "../result")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(
        {
            "epoch": list(range(1, len(history.get("train_loss", [])) + 1)),
            "train_loss": history.get("train_loss", []),
            "train_acc": history.get("train_acc", []),
            "val_loss": history.get("val_loss", []),
            "val_acc": history.get("val_acc", []),
        }
    )
    save_path = os.path.join(output_dir, f"training_history_fold_{cur}.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved training history to: {save_path}")

    # Per-fold best summary (greatest accuracy, smallest loss).
    if len(df) > 0:
        best_train_acc_idx = int(df["train_acc"].idxmax())
        best_val_acc_idx = int(df["val_acc"].idxmax())
        best_train_loss_idx = int(df["train_loss"].idxmin())
        best_val_loss_idx = int(df["val_loss"].idxmin())

        summary_row = {
            "fold": cur,
            "best_train_acc": float(df.loc[best_train_acc_idx, "train_acc"]),
            "best_train_acc_epoch": int(df.loc[best_train_acc_idx, "epoch"]),
            "best_val_acc": float(df.loc[best_val_acc_idx, "val_acc"]),
            "best_val_acc_epoch": int(df.loc[best_val_acc_idx, "epoch"]),
            "best_train_loss": float(df.loc[best_train_loss_idx, "train_loss"]),
            "best_train_loss_epoch": int(df.loc[best_train_loss_idx, "epoch"]),
            "best_val_loss": float(df.loc[best_val_loss_idx, "val_loss"]),
            "best_val_loss_epoch": int(df.loc[best_val_loss_idx, "epoch"]),
        }

        # Keep one row per fold in a shared summary file.
        summary_path = os.path.join(output_dir, "training_best_by_fold.csv")
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path)
            summary_df = summary_df[summary_df["fold"] != cur]
            summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
        else:
            summary_df = pd.DataFrame([summary_row])

        summary_df = summary_df.sort_values("fold").reset_index(drop=True)
        summary_df.to_csv(summary_path, index=False)
        print(
            f"[Fold {cur}] best_val_acc={summary_row['best_val_acc']:.4f} "
            f"(epoch {summary_row['best_val_acc_epoch']}), "
            f"best_val_loss={summary_row['best_val_loss']:.4f} "
            f"(epoch {summary_row['best_val_loss_epoch']})"
        )
        print(f"Saved per-fold best summary to: {summary_path}")

    return save_path