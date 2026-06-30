from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CURRENT_FILE = Path(__file__).resolve()


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "modelos").exists() and (parent / "data").exists():
            return parent
    raise RuntimeError("Não foi possível encontrar a raiz do projeto.")


ROOT = find_project_root(CURRENT_FILE)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Importa os avaliadores já existentes no projeto.
from modelos.cnn_baseline.evaluate import evaluate_checkpoint as evaluate_cnn
from modelos.mesonet.evaluate import evaluate_checkpoint as evaluate_meso
from modelos.resnext_lstm.evaluate import evaluate_run as evaluate_resnext


OUT = ROOT / "reports" / "matrizes_confusao"
OUT.mkdir(parents=True, exist_ok=True)


CONSOLIDADOS = {
    "CNN Baseline": ROOT / "reports" / "cnn_image" / "tables" / "consolidado" / "cnn_experimentos_consolidados.csv",
    "Meso-4": ROOT / "reports" / "mesonet_image" / "tables" / "consolidado" / "mesonet_experimentos_consolidados.csv",
    "ResNeXt+LSTM": ROOT / "reports" / "resnext_lstm" / "tables" / "consolidado" / "resnext_lstm_experimentos_consolidados.csv",
}


def normalizar_caminho(path_value: str | Path | None) -> Path | None:
    """
    Converte caminhos salvos no CSV para caminhos válidos no projeto atual.

    Isso é necessário porque alguns CSVs armazenam caminhos absolutos do Windows,
    como C:\\tcc mesonet\\experiments\\..., que podem quebrar se o projeto estiver
    em outra pasta. A função reaproveita o trecho a partir de 'experiments' ou
    'reports' e monta o caminho a partir da raiz atual do projeto.
    """
    if path_value is None or pd.isna(path_value):
        return None

    s = str(path_value).replace("\\", "/")
    p = Path(s)

    if p.exists():
        return p

    for marcador in ["experiments/", "reports/", "data/"]:
        if marcador in s:
            relativo = s.split(marcador, 1)[1]
            candidato = ROOT / marcador.rstrip("/") / relativo
            if candidato.exists():
                return candidato
            return candidato

    if not p.is_absolute():
        return ROOT / p

    return p


def carregar_melhor_linha(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo consolidado não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    if "test_f1_macro" not in df.columns:
        raise ValueError(f"O arquivo não possui a coluna test_f1_macro: {csv_path}")

    return df.sort_values("test_f1_macro", ascending=False).iloc[0]


def carregar_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def matriz_do_json(metrics: dict) -> tuple[np.ndarray, list[str]]:
    cm = np.array(metrics["confusion_matrix"], dtype=int)
    classes = metrics.get("classes", ["real", "fake"])
    return cm, classes


def avaliar_cnn(best: pd.Series) -> tuple[np.ndarray, list[str], dict]:
    model_path = normalizar_caminho(best.get("best_model_path"))
    output_dir = OUT / "cnn_baseline"

    metrics_file = normalizar_caminho(best.get("metrics_file"))
    if metrics_file and metrics_file.exists():
        metrics = carregar_json(metrics_file)
        if "confusion_matrix" in metrics:
            return (*matriz_do_json(metrics), metrics)

    metrics = evaluate_cnn(
        model_path=model_path,
        split_dir="data/splits/split_01_interval_10",
        img_size=int(best.get("img_size", 256)),
        batch_size=int(best.get("batch_size", 32)),
        dropout_fc=float(best.get("dropout_fc", 0.5)),
        output_dir=output_dir,
    )

    return (*matriz_do_json(metrics), metrics)


def avaliar_meso(best: pd.Series) -> tuple[np.ndarray, list[str], dict]:
    model_path = normalizar_caminho(best.get("best_model_path"))
    output_dir = OUT / "meso4"

    metrics_file = normalizar_caminho(best.get("metrics_file"))
    if metrics_file and metrics_file.exists():
        metrics = carregar_json(metrics_file)
        if "confusion_matrix" in metrics:
            return (*matriz_do_json(metrics), metrics)

    metrics = evaluate_meso(
        model_path=model_path,
        split_dir="data/splits/split_01_interval_10",
        img_size=int(best.get("img_size", 256)),
        batch_size=int(best.get("batch_size", 32)),
        dropout_conv=float(best.get("dropout_conv", 0.25)),
        dropout_fc=float(best.get("dropout_fc", 0.0)),
        output_dir=output_dir,
    )

    return (*matriz_do_json(metrics), metrics)


def avaliar_resnext(best: pd.Series) -> tuple[np.ndarray, list[str], dict]:
    run_dir = normalizar_caminho(best.get("run_dir"))
    metrics_file = normalizar_caminho(best.get("metrics_file"))

    if metrics_file and metrics_file.exists():
        metrics = carregar_json(metrics_file)
        if "confusion_matrix" in metrics:
            return (*matriz_do_json(metrics), metrics)

    # Caso a coluna confusion_matrix exista no CSV consolidado, usa diretamente.
    if "confusion_matrix" in best and pd.notna(best.get("confusion_matrix")):
        cm = np.array(ast.literal_eval(str(best["confusion_matrix"])), dtype=int)
        classes = ["real", "fake"]
        metrics = {
            "test_accuracy": float(best.get("test_accuracy", 0)),
            "test_precision_macro": float(best.get("test_precision_macro", 0)),
            "test_recall_macro": float(best.get("test_recall_macro", 0)),
            "test_f1_macro": float(best.get("test_f1_macro", 0)),
            "classes": classes,
            "confusion_matrix": cm.tolist(),
        }
        return cm, classes, metrics

    if run_dir is None:
        raise FileNotFoundError("Não foi possível localizar o run_dir da ResNeXt+LSTM.")

    metrics = evaluate_resnext(run_dir)
    return (*matriz_do_json(metrics), metrics)


def estilizar_eixos(ax, classes: list[str], titulo: str):
    ax.set_title(titulo, fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel("Previsto", fontsize=16, fontweight="bold")
    ax.set_ylabel("Real", fontsize=16, fontweight="bold")

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=15, fontweight="bold")

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=15, fontweight="bold")

    ax.tick_params(axis="both", width=1.4)


def plotar_matriz(ax, cm: np.ndarray, classes: list[str], titulo: str):
    im = ax.imshow(cm, cmap="Blues")

    estilizar_eixos(ax, classes, titulo)

    limite = cm.max() / 2 if cm.size else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            valor = int(cm[i, j])
            cor = "white" if valor > limite else "black"
            ax.text(
                j,
                i,
                str(valor),
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
                color=cor,
            )

    return im


def salvar_matriz_individual(nome: str, cm: np.ndarray, classes: list[str], titulo: str):
    fig, ax = plt.subplots(figsize=(7, 6))

    im = plotar_matriz(ax, cm, classes, titulo)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=13)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(OUT / f"matriz_confusao_{nome}.png", dpi=500, bbox_inches="tight")
    plt.close(fig)


def salvar_matriz_comparativa(resultados: dict):
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.5))

    for ax, (modelo, dados) in zip(axs, resultados.items()):
        cm = dados["cm"]
        classes = dados["classes"]
        plotar_matriz(ax, cm, classes, modelo)

    fig.tight_layout(w_pad=3.0)
    fig.savefig(OUT / "matrizes_confusao_arquiteturas.png", dpi=500, bbox_inches="tight")
    plt.close(fig)


def salvar_resumo(resultados: dict):
    linhas = []

    for modelo, dados in resultados.items():
        metrics = dados["metrics"]
        cm = dados["cm"]

        linhas.append({
            "Modelo": modelo,
            "Matriz": cm.tolist(),
            "Acurácia": metrics.get("test_accuracy", metrics.get("accuracy")),
            "Precisão macro": metrics.get("test_precision_macro", metrics.get("precision_macro")),
            "Recall macro": metrics.get("test_recall_macro", metrics.get("recall_macro")),
            "F1 macro": metrics.get("test_f1_macro", metrics.get("f1_macro")),
        })

    df = pd.DataFrame(linhas)
    df.to_csv(OUT / "resumo_matrizes_confusao.csv", index=False, encoding="utf-8")


def main():
    print("=" * 70)
    print("GERAÇÃO DAS MATRIZES DE CONFUSÃO GERAIS")
    print("=" * 70)

    resultados = {}

    best_cnn = carregar_melhor_linha(CONSOLIDADOS["CNN Baseline"])
    cm, classes, metrics = avaliar_cnn(best_cnn)
    resultados["CNN Baseline"] = {"cm": cm, "classes": classes, "metrics": metrics}
    salvar_matriz_individual("cnn_baseline", cm, classes, "CNN Baseline")
    print(f"[OK] CNN Baseline | matriz={cm.tolist()}")

    best_meso = carregar_melhor_linha(CONSOLIDADOS["Meso-4"])
    cm, classes, metrics = avaliar_meso(best_meso)
    resultados["Meso-4"] = {"cm": cm, "classes": classes, "metrics": metrics}
    salvar_matriz_individual("meso4", cm, classes, "Meso-4")
    print(f"[OK] Meso-4 | matriz={cm.tolist()}")

    best_resnext = carregar_melhor_linha(CONSOLIDADOS["ResNeXt+LSTM"])
    cm, classes, metrics = avaliar_resnext(best_resnext)
    resultados["ResNeXt+LSTM"] = {"cm": cm, "classes": classes, "metrics": metrics}
    salvar_matriz_individual("resnext_lstm", cm, classes, "ResNeXt+LSTM")
    print(f"[OK] ResNeXt+LSTM | matriz={cm.tolist()}")

    salvar_matriz_comparativa(resultados)
    salvar_resumo(resultados)

    print("=" * 70)
    print(f"Arquivos gerados em: {OUT}")
    print("Imagem comparativa:", OUT / "matrizes_confusao_arquiteturas.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
