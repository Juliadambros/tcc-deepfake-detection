from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_from_root(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_barplot(df, x, y, title, output_path, figsize=(10, 6), rotation=45):
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(df[x].astype(str), df[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=rotation)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_lineplot(df, x, y, title, output_path, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    plot_df = df.sort_values(x)
    ax.plot(plot_df[x], plot_df[y], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_grouped_barplot(df, category_col, series_col, value_col, title, output_path, figsize=(10, 6)):
    pivot_df = df.pivot_table(
        index=category_col,
        columns=series_col,
        values=value_col,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=figsize)
    pivot_df.plot(kind="bar", ax=ax)

    ax.set_title(title)
    ax.set_xlabel(category_col)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_scatterplot(df, x, y, title, output_path, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(df[x], df[y], alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_top10_time_plot(df, output_path, figsize=(13, 7)):
    plot_df = df.copy()

    plot_df["rank_num"] = (
        plot_df["rank"].str.replace("Exp ", "", regex=False).astype(int)
    )

    plot_df = plot_df.sort_values("rank_num")

    labels = [
        f"{r} ({int(e)} ep.)"
        for r, e in zip(plot_df["rank"], plot_df["epochs_executed"])
    ]

    tempos = plot_df["total_training_time_sec"].fillna(0)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, tempos)

    ax.invert_yaxis()
    ax.set_title(
        "Tempo de treinamento dos 10 melhores experimentos\n"
        "(ordenados por F1-macro no teste)",
        fontsize=15,
        pad=15,
    )

    ax.set_xlabel("Tempo total de treinamento (segundos)")
    ax.set_ylabel("Experimentos")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, valor in zip(bars, tempos):
        ax.text(
            valor + 25,
            bar.get_y() + bar.get_height() / 2,
            f"{int(valor)} s",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Gera análise resumida do CSV consolidado da ResNeXt+LSTM."
    )

    parser.add_argument(
        "--csv",
        default="reports/resnext_lstm/tables/consolidado/resnext_lstm_experimentos_consolidados.csv",
    )

    parser.add_argument(
        "--out-dir",
        default="reports/resnext_lstm/figures/consolidado",
    )

    parser.add_argument(
        "--ignore-lr",
        nargs="*",
        type=float,
        default=[0.01],
    )

    args = parser.parse_args()

    csv_path = resolve_from_root(args.csv)
    out_dir = ensure_dir(resolve_from_root(args.out_dir))

    df = pd.read_csv(csv_path)

    if df.empty:
        print("CSV vazio.")
        return

    if "learning_rate" in df.columns and args.ignore_lr:
        df = df[~df["learning_rate"].isin(args.ignore_lr)].copy()

    required_cols = [
        "run_name",
        "epochs_requested",
        "epochs_executed",
        "batch_size",
        "seq_len",
        "learning_rate",
        "optimizer",
        "dropout",
        "weight_decay",
        "test_f1_macro",
        "test_accuracy",
        "best_epoch",
        "total_training_time_sec",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    df_sorted = df.sort_values("test_f1_macro", ascending=False).reset_index(drop=True)

    ranking = df_sorted.head(10).copy()
    ranking["rank"] = [f"Exp {i+1}" for i in range(len(ranking))]

    ranking_path = out_dir / "top_10_experimentos.csv"
    ranking.to_csv(ranking_path, index=False, encoding="utf-8")

    save_barplot(
        ranking,
        x="rank",
        y="test_f1_macro",
        title="Top 10 experimentos ResNeXt+LSTM por F1-macro no teste",
        output_path=out_dir / "top10_f1_macro.png",
        figsize=(11, 6),
        rotation=0,
    )

    save_barplot(
        ranking,
        x="rank",
        y="test_accuracy",
        title="Top 10 experimentos ResNeXt+LSTM por acurácia no teste",
        output_path=out_dir / "top10_accuracy.png",
        figsize=(11, 6),
        rotation=0,
    )

    save_top10_time_plot(
        ranking,
        output_path=out_dir / "top10_tempo_epocas.png",
    )

    mean_by_epochs_requested = (
        df.groupby("epochs_requested", as_index=False)["test_f1_macro"]
        .mean()
        .sort_values("epochs_requested")
    )

    mean_by_epochs_requested.to_csv(
        out_dir / "media_f1_por_epocas_solicitadas.csv",
        index=False,
        encoding="utf-8",
    )

    save_lineplot(
        mean_by_epochs_requested,
        x="epochs_requested",
        y="test_f1_macro",
        title="F1-macro médio por épocas solicitadas",
        output_path=out_dir / "f1_por_epocas_solicitadas.png",
    )

    mean_by_epochs_executed = (
        df.groupby("epochs_executed", as_index=False)["test_f1_macro"]
        .mean()
        .sort_values("epochs_executed")
    )

    mean_by_epochs_executed.to_csv(
        out_dir / "media_f1_por_epocas_executadas.csv",
        index=False,
        encoding="utf-8",
    )

    save_lineplot(
        mean_by_epochs_executed,
        x="epochs_executed",
        y="test_f1_macro",
        title="F1-macro médio por épocas executadas",
        output_path=out_dir / "f1_por_epocas_executadas.png",
    )

    mean_by_optimizer = (
        df.groupby("optimizer", as_index=False)["test_f1_macro"]
        .mean()
        .sort_values("test_f1_macro", ascending=False)
    )

    mean_by_optimizer.to_csv(
        out_dir / "media_f1_por_optimizer.csv",
        index=False,
        encoding="utf-8",
    )

    save_barplot(
        mean_by_optimizer,
        x="optimizer",
        y="test_f1_macro",
        title="F1-macro médio por otimizador",
        output_path=out_dir / "f1_por_optimizer.png",
        rotation=0,
    )

    mean_by_optimizer_lr = (
        df.groupby(["optimizer", "learning_rate"], as_index=False)["test_f1_macro"]
        .mean()
        .sort_values(["optimizer", "learning_rate"])
    )

    mean_by_optimizer_lr.to_csv(
        out_dir / "media_f1_por_optimizer_learning_rate.csv",
        index=False,
        encoding="utf-8",
    )

    save_grouped_barplot(
        mean_by_optimizer_lr,
        category_col="optimizer",
        series_col="learning_rate",
        value_col="test_f1_macro",
        title="F1-macro médio por otimizador e learning rate",
        output_path=out_dir / "f1_por_optimizer_learning_rate.png",
    )

    mean_by_dropout = (
        df.groupby("dropout", as_index=False)["test_f1_macro"]
        .mean()
        .sort_values("dropout")
    )

    mean_by_dropout.to_csv(
        out_dir / "media_f1_por_dropout.csv",
        index=False,
        encoding="utf-8",
    )

    save_barplot(
        mean_by_dropout,
        x="dropout",
        y="test_f1_macro",
        title="F1-macro médio por dropout",
        output_path=out_dir / "f1_por_dropout.png",
        rotation=0,
    )

    mean_by_weight_decay = (
        df.groupby("weight_decay", as_index=False)["test_f1_macro"]
        .mean()
        .sort_values("weight_decay")
    )

    mean_by_weight_decay.to_csv(
        out_dir / "media_f1_por_weight_decay.csv",
        index=False,
        encoding="utf-8",
    )

    save_barplot(
        mean_by_weight_decay,
        x="weight_decay",
        y="test_f1_macro",
        title="F1-macro médio por weight_decay",
        output_path=out_dir / "f1_por_weight_decay.png",
        rotation=45,
    )

    mean_by_seq_len = (
        df.groupby("seq_len", as_index=False)["test_f1_macro"]
        .mean()
        .sort_values("seq_len")
    )

    mean_by_seq_len.to_csv(
        out_dir / "media_f1_por_seq_len.csv",
        index=False,
        encoding="utf-8",
    )

    save_barplot(
        mean_by_seq_len,
        x="seq_len",
        y="test_f1_macro",
        title="F1-macro médio por tamanho da sequência",
        output_path=out_dir / "f1_por_seq_len.png",
        rotation=0,
    )

    save_scatterplot(
        df,
        x="best_epoch",
        y="test_f1_macro",
        title="Relação entre melhor época e F1-macro no teste",
        output_path=out_dir / "scatter_best_epoch_vs_f1.png",
    )

    best = df_sorted.iloc[0].to_dict()

    with open(out_dir / "melhor_experimento.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    summary_lines = [
        "Resumo da análise ResNeXt+LSTM",
        "==============================",
        f"Total de experimentos: {len(df)}",
        "",
        "Melhor experimento:",
        f"- run_name: {best['run_name']}",
        f"- test_f1_macro: {best['test_f1_macro']:.6f}",
        f"- test_accuracy: {best['test_accuracy']:.6f}",
        f"- optimizer: {best['optimizer']}",
        f"- learning_rate: {best['learning_rate']}",
        f"- batch_size: {best['batch_size']}",
        f"- seq_len: {best['seq_len']}",
        f"- epochs_requested: {best['epochs_requested']}",
        f"- epochs_executed: {best['epochs_executed']}",
        f"- best_epoch: {best['best_epoch']}",
        f"- dropout: {best['dropout']}",
        f"- weight_decay: {best['weight_decay']}",
        f"- total_training_time_sec: {best['total_training_time_sec']}",
    ]

    with open(out_dir / "resumo_analise.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("Análise concluída.")
    print(f"Top 10 salvo em: {ranking_path}")
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()