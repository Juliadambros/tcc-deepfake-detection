from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def main():
    parser = argparse.ArgumentParser(
        description="Gera métricas (ROC, PR, histogramas) a partir do CSV de vídeos."
    )

    parser.add_argument("--csv", required=True, help="Caminho do CSV de resultados")
    parser.add_argument("--out", required=True, help="Pasta de saída dos gráficos")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    df["y_true"] = df["true_class"].map({"fake": 1, "real": 0})
    df["y_score"] = df["prob_fake"]

    fpr, tpr, _ = roc_curve(df["y_true"], df["y_score"])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Vídeos)")
    plt.legend()
    plt.grid()
    plt.savefig(out_dir / "roc_curve.png", dpi=200)
    plt.close()

    precision, recall, _ = precision_recall_curve(df["y_true"], df["y_score"])

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Vídeos)")
    plt.grid()
    plt.savefig(out_dir / "precision_recall.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(df[df["y_true"] == 1]["y_score"], bins=20, alpha=0.6, label="Fake")
    plt.hist(df[df["y_true"] == 0]["y_score"], bins=20, alpha=0.6, label="Real")
    plt.xlabel("Probabilidade de ser Fake")
    plt.ylabel("Frequência")
    plt.title("Distribuição das Probabilidades")
    plt.legend()
    plt.grid()
    plt.savefig(out_dir / "histograma_probabilidades.png", dpi=200)
    plt.close()

    plt.figure()
    data = [
        df[df["y_true"] == 1]["y_score"],
        df[df["y_true"] == 0]["y_score"],
    ]
    plt.boxplot(data, labels=["Fake", "Real"])
    plt.ylabel("Probabilidade de ser Fake")
    plt.title("Boxplot das Probabilidades")
    plt.grid()
    plt.savefig(out_dir / "boxplot.png", dpi=200)
    plt.close()

    df["confidence"] = np.abs(df["prob_fake"] - df["prob_real"])

    mean_conf = df["confidence"].mean()
    min_conf = df["confidence"].min()

    hard_cases = df.sort_values("confidence").head(10)

    with open(out_dir / "resumo_metricas.txt", "w", encoding="utf-8") as f:
        f.write("=== ANÁLISE DE MÉTRICAS ===\n\n")
        f.write(f"AUC: {roc_auc:.4f}\n")
        f.write(f"Confiança média: {mean_conf:.4f}\n")
        f.write(f"Menor confiança: {min_conf:.4f}\n\n")

        f.write("=== CASOS MAIS DIFÍCEIS ===\n")
        f.write(hard_cases.to_string(index=False))

    print("\nAnálises geradas com sucesso!")
    print(f"Pasta: {out_dir}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Confiança média: {mean_conf:.4f}")


if __name__ == "__main__":
    main()