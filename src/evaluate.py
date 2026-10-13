import sys
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent))
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import RESULTS_DIR

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
    }


def plot_class_distribution(
    labels: List[int],
    class_names: List[str],
    save_path: Path = RESULTS_DIR / "fig3_1_class_distribution.png",
) -> None:
    counts = pd.Series(labels).value_counts().sort_index()
    short_names = [name.replace("___", " - ").replace("_", " ") for name in class_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = sns.barplot(x=counts.values, y=short_names, hue=short_names, palette="viridis", legend=False, ax=ax, orient="h")
    ax.set_title("Hình 3.1: Phân phối số lượng ảnh theo các lớp bệnh lá cây (EDA)")
    ax.set_xlabel("Số lượng mẫu ảnh (Ảnh)")
    ax.set_ylabel("Lớp bệnh / Loại cây")

    for bar in bars.patches:
        width = int(bar.get_width())
        ax.text(
            width + max(counts.values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width}",
            va="center",
            fontsize=10,
        )

    ax.set_xlim(0, max(counts.values) * 1.12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    title: str = "Hình 4.1: Ma trận nhầm lẫn (Confusion Matrix) trên tập Test",
    save_path: Path = RESULTS_DIR / "fig4_1_confusion_matrix.png",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    short_names = [name.replace("___", "\n").replace("_", " ") for name in class_names]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        cbar_kws={"label": "Số lượng mẫu (Ảnh)"},
        ax=ax,
    )
    ax.set_title(title, pad=14)
    ax.set_xlabel("Nhãn dự đoán (Predicted Label)")
    ax.set_ylabel("Nhãn thực tế (Actual Label)")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_learning_curves(
    history_df: pd.DataFrame,
    save_path: Path = RESULTS_DIR / "fig4_2_learning_curves.png",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train Loss", linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], marker="s", label="Validation Loss", linewidth=2)
    axes[0].set_title("Biến thiên hàm mất mát (Loss) theo Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_f1"], marker="o", label="Train Macro F1", linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_f1"], marker="s", label="Validation Macro F1", linewidth=2)
    axes[1].set_title("Biến thiên Macro F1-Score theo Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1-Score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    fig.suptitle("Hình 4.2: Đường học (Learning Curves) theo dõi hiện tượng Overfitting", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def analyze_misclassified_samples(
    image_paths: List[Path],
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[np.ndarray],
    class_names: List[str],
    max_samples: int = 10,
) -> pd.DataFrame:
    records = []
    for idx, (path, true_idx, pred_idx) in enumerate(zip(image_paths, y_true, y_pred)):
        if true_idx != pred_idx:
            confidence = float(y_prob[idx, pred_idx]) if y_prob is not None else None
            true_prob = float(y_prob[idx, true_idx]) if y_prob is not None else None
            records.append({
                "image_path": str(path.name),
                "full_path": str(path),
                "actual_class": class_names[true_idx],
                "predicted_class": class_names[pred_idx],
                "pred_confidence": round(confidence, 4) if confidence is not None else "N/A",
                "actual_class_prob": round(true_prob, 4) if true_prob is not None else "N/A",
            })

    df_errors = pd.DataFrame(records)
    if not df_errors.empty:
        df_errors.to_csv(RESULTS_DIR / "misclassified_samples.csv", index=False)
    return df_errors.head(max_samples)
