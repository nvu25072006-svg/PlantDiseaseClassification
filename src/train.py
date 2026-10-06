import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    EARLY_STOPPING_PATIENCE,
    RESULTS_DIR,
    SAMPLE_CLASSES,
    SEED,
    SPLITS_DIR,
    WEIGHT_DECAY,
    seed_everything,
)
from src.data import (
    PlantLeafDataset,
    build_tabular_features,
    get_transforms,
    materialize_splits_to_disk,
    scan_image_dataset,
    stratified_holdout_split,
)
from src.evaluate import (
    analyze_misclassified_samples,
    compute_classification_metrics,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_learning_curves,
)
from src.model import (
    create_color_logreg_pipeline,
    create_deep_model,
    create_dummy_baseline,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    torch = None


def log_experiment_row(
    exp_list: List[Dict],
    exp_id: int,
    model_name: str,
    config_desc: str,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float] = None,
    elapsed_sec: float = 0.0,
    note: str = "",
) -> None:
    row = {
        "exp_id": exp_id,
        "model": model_name,
        "config": config_desc,
        "val_acc": val_metrics["accuracy"],
        "val_precision": val_metrics["precision_macro"],
        "val_recall": val_metrics["recall_macro"],
        "val_f1": val_metrics["f1_macro"],
        "test_acc": test_metrics["accuracy"] if test_metrics else None,
        "test_precision": test_metrics["precision_macro"] if test_metrics else None,
        "test_recall": test_metrics["recall_macro"] if test_metrics else None,
        "test_f1": test_metrics["f1_macro"] if test_metrics else None,
        "time_sec": round(elapsed_sec, 2),
        "note": note,
    }
    exp_list.append(row)


def train_and_eval_pytorch_model(
    arch_name: str,
    splits: Dict[str, Tuple[List[Path], List[int]]],
    num_classes: int,
    class_names: List[str],
    epochs: int = 8,
    lr: float = 3e-4,
    freeze_backbone: bool = False,
    use_augmentation: bool = True,
    save_checkpoint_name: str = "best_model.pth",
):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, eval_tf = get_transforms(use_augmentation=use_augmentation)
    train_ds = PlantLeafDataset(*splits["train"], transform=train_tf)
    val_ds = PlantLeafDataset(*splits["val"], transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = create_deep_model(
        arch_name=arch_name,
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=freeze_backbone,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_f1 = -1.0
    best_val_metrics = {}
    patience_counter = 0
    best_ckpt_path = RESULTS_DIR / save_checkpoint_name

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().tolist()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().tolist())

        train_loss = running_loss / len(train_ds)
        train_metrics = compute_classification_metrics(train_targets, train_preds)

        model.eval()
        val_running_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = torch.as_tensor(labels, dtype=torch.long, device=device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1).cpu().tolist()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().tolist())

        val_loss = val_running_loss / len(val_ds)
        val_metrics = compute_classification_metrics(val_targets, val_preds)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_f1": train_metrics["f1_macro"],
            "val_f1": val_metrics["f1_macro"],
            "val_acc": val_metrics["accuracy"],
        })

        print(
            f"  [Epoch {epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_metrics['f1_macro']:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_val_metrics = val_metrics
            patience_counter = 0
            torch.save(
                {
                    "arch_name": arch_name,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "state_dict": model.state_dict(),
                    "val_metrics": best_val_metrics,
                    "epoch": epoch,
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test_ds = PlantLeafDataset(*splits["test"], transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().tolist()
            test_preds.extend(preds)
            test_targets.extend(labels.tolist())
            test_probs.append(probs)
    test_probs_arr = np.vstack(test_probs)
    test_metrics = compute_classification_metrics(test_targets, test_preds)

    history_df = pd.DataFrame(history)
    return best_val_metrics, test_metrics, history_df, best_ckpt_path, test_preds, test_probs_arr


def run_all_experiments(
    epochs: int = 6,
    use_all_38_classes: bool = False,
    max_per_class: int = 300,
) -> None:
    seed_everything(SEED)
    selected = None if use_all_38_classes else SAMPLE_CLASSES
    image_paths, labels, class_names = scan_image_dataset(
        DATA_DIR,
        selected_classes=selected,
        max_per_class=max_per_class,
    )
    raw_splits = stratified_holdout_split(image_paths, labels)
    splits = materialize_splits_to_disk(raw_splits, class_names, splits_dir=SPLITS_DIR)

    train_paths, y_train = splits["train"]
    val_paths, y_val = splits["val"]
    test_paths, y_test = splits["test"]

    plot_class_distribution(labels, class_names)

    X_train_tab = build_tabular_features(train_paths)
    X_val_tab = build_tabular_features(val_paths)
    X_test_tab = build_tabular_features(test_paths)

    experiment_rows: List[Dict] = []

    t0 = time.time()
    dummy_clf = create_dummy_baseline(strategy="most_frequent")
    dummy_clf.fit(X_train_tab, y_train)
    val_pred_dummy = dummy_clf.predict(X_val_tab)
    test_pred_dummy = dummy_clf.predict(X_test_tab)
    val_metrics_dummy = compute_classification_metrics(y_val, val_pred_dummy)
    test_metrics_dummy = compute_classification_metrics(y_test, test_pred_dummy)
    log_experiment_row(
        experiment_rows,
        exp_id=1,
        model_name="Dummy (lớp đa số)",
        config_desc="strategy='most_frequent'",
        val_metrics=val_metrics_dummy,
        test_metrics=test_metrics_dummy,
        elapsed_sec=time.time() - t0,
        note="baseline",
    )

    t0 = time.time()
    color_logreg = create_color_logreg_pipeline(c_param=1.0)
    color_logreg.fit(X_train_tab, y_train)
    val_pred_lr = color_logreg.predict(X_val_tab)
    test_pred_lr = color_logreg.predict(X_test_tab)
    val_prob_lr = color_logreg.predict_proba(X_test_tab)
    val_metrics_lr = compute_classification_metrics(y_val, val_pred_lr)
    test_metrics_lr = compute_classification_metrics(y_test, test_pred_lr)
    joblib.dump(
        {"pipeline": color_logreg, "class_names": class_names},
        RESULTS_DIR / "color_logreg_pipeline.joblib",
    )
    log_experiment_row(
        experiment_rows,
        exp_id=2,
        model_name="HSV Color Hist + Logistic Regression",
        config_desc="HSV 8-bins + StandardScaler + LogReg (C=1.0)",
        val_metrics=val_metrics_lr,
        test_metrics=test_metrics_lr,
        elapsed_sec=time.time() - t0,
        note="classic ML baseline",
    )

    plot_confusion_matrix(y_test, test_pred_lr, class_names)
    analyze_misclassified_samples(test_paths, y_test, test_pred_lr, val_prob_lr, class_names)

    if torch is not None:
        t0 = time.time()
        val_m_mobilenet, test_m_mobilenet, _, _, _, _ = train_and_eval_pytorch_model(
            arch_name="mobilenetv3_small_100",
            splits=splits,
            num_classes=len(class_names),
            class_names=class_names,
            epochs=epochs,
            lr=1e-3,
            freeze_backbone=True,
            use_augmentation=True,
            save_checkpoint_name="mobilenetv3_frozen.pth",
        )
        log_experiment_row(
            experiment_rows,
            exp_id=3,
            model_name="MobileNetV3-Small (frozen)",
            config_desc=f"freeze backbone, lr=1e-3, {epochs} epochs, batch={BATCH_SIZE}",
            val_metrics=val_m_mobilenet,
            test_metrics=test_m_mobilenet,
            elapsed_sec=time.time() - t0,
            note="frozen backbone",
        )

        t0 = time.time()
        val_m_resnet, test_m_resnet, history_df, _, resnet_test_preds, resnet_test_probs = train_and_eval_pytorch_model(
            arch_name="resnet18",
            splits=splits,
            num_classes=len(class_names),
            class_names=class_names,
            epochs=epochs,
            lr=3e-4,
            freeze_backbone=False,
            use_augmentation=True,
            save_checkpoint_name="best_model.pth",
        )
        history_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)
        plot_learning_curves(history_df)
        plot_confusion_matrix(y_test, resnet_test_preds, class_names)
        analyze_misclassified_samples(test_paths, y_test, resnet_test_preds, resnet_test_probs, class_names)
        log_experiment_row(
            experiment_rows,
            exp_id=4,
            model_name="ResNet18 (fine-tune)",
            config_desc=f"unfreeze all, lr=3e-4, Aug=True, {epochs} epochs",
            val_metrics=val_m_resnet,
            test_metrics=test_m_resnet,
            elapsed_sec=time.time() - t0,
            note="main model",
        )

    exp_df = pd.DataFrame(experiment_rows)
    exp_df.to_csv(RESULTS_DIR / "experiment_log.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--all_classes", action="store_true")
    parser.add_argument("--max_per_class", type=int, default=300)
    args = parser.parse_args()
    run_all_experiments(
        epochs=args.epochs,
        use_all_38_classes=args.all_classes,
        max_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
