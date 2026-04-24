from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_metrics_file(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["metrics_file"] = str(path)
        data["experiment_group"] = path.parents[1].name
        data["run_folder"] = path.parent.name
        return data

    except Exception as e:
        print(f"[AVISO] Não foi possível ler {path}: {e}")
        return None


def collect_all_metrics(experiments_root: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for metrics_path in experiments_root.rglob("metrics.json"):
        row = load_metrics_file(metrics_path)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "test_f1_macro" in df.columns:
        df = df.sort_values("test_f1_macro", ascending=False).reset_index(drop=True)

    return df


def save_summary_files(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    consolidated_csv = out_dir / "cnn_experimentos_consolidados.csv"
    df.to_csv(consolidated_csv, index=False, encoding="utf-8")

    if df.empty:
        print("Nenhum experimento encontrado.")
        return

    best = df.iloc[0].to_dict()

    with open(out_dir / "melhor_experimento_geral.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    df.head(10).to_csv(
        out_dir / "top_10_experimentos_geral.csv",
        index=False,
        encoding="utf-8",
    )

    cols_to_group = [
        "experiment_group",
        "epochs_requested",
        "epochs_executed",
        "optimizer",
        "learning_rate",
        "batch_size",
        "dropout_fc",
        "weight_decay",
    ]

    for col in cols_to_group:
        if col in df.columns:
            grouped = (
                df.groupby(col, as_index=False)[
                    ["test_f1_macro", "test_accuracy", "best_val_acc"]
                ]
                .mean(numeric_only=True)
                .sort_values("test_f1_macro", ascending=False)
            )
            grouped.to_csv(
                out_dir / f"media_por_{col}.csv",
                index=False,
                encoding="utf-8",
            )

    lines = [
        "Resumo consolidado dos experimentos de imagens CNN baseline",
        "==========================================================",
        f"Total de experimentos encontrados: {len(df)}",
        "",
        "Melhor experimento geral:",
        f"- run_name: {best.get('run_name', '')}",
        f"- experiment_group: {best.get('experiment_group', '')}",
        f"- test_f1_macro: {best.get('test_f1_macro', '')}",
        f"- test_accuracy: {best.get('test_accuracy', '')}",
        f"- best_val_acc: {best.get('best_val_acc', '')}",
        f"- best_epoch: {best.get('best_epoch', '')}",
        f"- optimizer: {best.get('optimizer', '')}",
        f"- learning_rate: {best.get('learning_rate', '')}",
        f"- batch_size: {best.get('batch_size', '')}",
        f"- dropout_fc: {best.get('dropout_fc', '')}",
        f"- weight_decay: {best.get('weight_decay', '')}",
        f"- epochs_requested: {best.get('epochs_requested', '')}",
        f"- run_dir: {best.get('run_dir', '')}",
    ]

    with open(out_dir / "resumo_consolidado.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Consolidação concluída. Arquivos salvos em: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolida todos os experiments/cnn_baseline em um único CSV."
    )
    parser.add_argument(
        "--experiments-root",
        default="experiments/cnn_baseline",
        help="Pasta raiz dos experimentos da CNN baseline.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/cnn/tables/consolidado",
        help="Pasta de saída dos arquivos consolidados.",
    )
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not experiments_root.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {experiments_root}")

    df = collect_all_metrics(experiments_root)
    save_summary_files(df, out_dir)


if __name__ == "__main__":
    main()