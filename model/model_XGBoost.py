import os
import pickle

import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


def _to_numpy(split_dataset):
    return split_dataset.x, split_dataset.y, split_dataset.df


def _extract_case_id(row, default_idx):
    for key in ("patient_id", "_PATIENT", "sampleID", "bcr_patient_barcode"):
        if key in row and row[key] is not None:
            return str(row[key])
    return f"sample_{default_idx}"


def _to_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_event_time(row):
    for key in ("CDE_survival_time", "survival_months"):
        if key in row:
            return _to_float(row[key], 0.0)
    return 0.0


def _get_censorship(row):
    for key in ("censorship", "censor", "event", "status"):
        if key in row:
            return _to_float(row[key], 0.0)
    return 0.0


def _accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def _log_loss(y_true, y_prob, num_classes):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if num_classes == 2 and y_prob.ndim == 2 and y_prob.shape[1] == 2:
        true_probs = y_prob[np.arange(len(y_true)), y_true]
        return float(-np.mean(np.log(true_probs)))

    one_hot = np.eye(num_classes, dtype=np.float64)[y_true]
    return float(-np.mean(np.sum(one_hot * np.log(y_prob), axis=1)))


def train_xgboost_fold(args, train_dataset, test_dataset, cur, result_dir):
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is not installed. Install it with `pip install xgboost` "
            "or add it to your environment before using `--modality xgboost`."
        ) from XGBOOST_IMPORT_ERROR

    x_train, y_train, _ = _to_numpy(train_dataset)
    x_val, y_val, val_df = _to_numpy(test_dataset)

    num_classes = int(args.n_classes)
    learning_rate = getattr(args, "xgb_learning_rate", None)
    if learning_rate is None:
        learning_rate = args.lr
    model = XGBClassifier(
        objective="multi:softprob" if num_classes > 2 else "binary:logistic",
        num_class=num_classes if num_classes > 2 else None,
        n_estimators=int(getattr(args, "xgb_n_estimators", 300)),
        max_depth=int(getattr(args, "xgb_max_depth", 6)),
        learning_rate=float(learning_rate),
        subsample=float(getattr(args, "xgb_subsample", 1.0)),
        colsample_bytree=float(getattr(args, "xgb_colsample_bytree", 1.0)),
        reg_lambda=float(getattr(args, "xgb_reg_lambda", 1.0)),
        min_child_weight=float(getattr(args, "xgb_min_child_weight", 1.0)),
        random_state=int(args.seed),
        tree_method=getattr(args, "xgb_tree_method", "auto"),
        eval_metric="mlogloss" if num_classes > 2 else "logloss",
    )
    model.fit(x_train, y_train)

    train_probs = model.predict_proba(x_train)
    train_preds = np.argmax(train_probs, axis=1) if num_classes > 2 else (train_probs[:, 1] >= 0.5).astype(np.int64)
    val_probs = model.predict_proba(x_val)
    val_preds = np.argmax(val_probs, axis=1) if num_classes > 2 else (val_probs[:, 1] >= 0.5).astype(np.int64)

    train_acc = _accuracy_score(y_train, train_preds)
    val_acc = _accuracy_score(y_val, val_preds)
    train_loss = _log_loss(y_train, train_probs, num_classes)
    val_loss = _log_loss(y_val, val_probs, num_classes)

    checkpoint_dir = os.path.join(result_dir, "model_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"best_model_fold_{cur}.pkl")
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)

    patient_results = {}
    for idx, (_, row) in enumerate(val_df.reset_index(drop=True).iterrows()):
        row_dict = row.to_dict()
        case_id = _extract_case_id(row_dict, idx)
        patient_results[case_id] = {
            "label": int(y_val[idx]),
            "prediction": int(val_preds[idx]),
            "correct": bool(val_preds[idx] == y_val[idx]),
            "event_time": _get_event_time(row_dict),
            "censorship": _get_censorship(row_dict),
            "clinical": row_dict,
            "probabilities": val_probs[idx],
        }

    results_path = os.path.join(checkpoint_dir, f"split_{cur}_results.pkl")
    with open(results_path, "wb") as handle:
        pickle.dump(patient_results, handle)

    history = {
        "train_loss": [train_loss],
        "train_acc": [train_acc],
    }
    print(f"[Fold {cur}] XGBoost model saved to: {model_path}")
    print(f"[Fold {cur}] Final evaluation saved to: {results_path}")
    print(f"[Fold {cur}] Final val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")

    return history, (train_acc, train_loss, val_acc, val_loss), model_path, patient_results
