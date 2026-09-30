"""
Funcoes comuns aos scripts 11-13 (Samuel).

A avaliacao replica EXATAMENTE o protocolo do baseline trivial (script 05):
  1. metricas calculadas por (fold, alvo) -> 25 alvos x 5 folds;
  2. media entre os 25 alvos dentro de cada fold;
  3. media +/- desvio-padrao entre os 5 folds.
Assim os numeros dos modelos sao diretamente comparaveis aos do baseline.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
ROI_DIR = ROOT / "data" / "interim" / "rois"

SEED = 42
N_SPLITS = 5
LABELS = ["Normal/Mild", "Moderate", "Severe"]
LABEL_TO_INT = {lab: i for i, lab in enumerate(LABELS)}
# Pesos da metrica oficial da competicao RSNA 2024 (log-loss ponderada)
CLASS_WEIGHTS_OFICIAIS = {"Normal/Mild": 1.0, "Moderate": 2.0, "Severe": 4.0}

CONDITIONS = [
    "Spinal Canal Stenosis",
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
]
LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
ID_COLS = ["study_id", "series_id", "instance_number", "condition", "level"]


def target_name(condition, level):
    """'Spinal Canal Stenosis' + 'L1/L2' -> 'spinal_canal_stenosis_l1_l2'."""
    return (condition.lower().replace(" ", "_") + "_"
            + level.lower().replace("/", "_"))


def carregar_rotulos_longos():
    """sample_train.csv (formato largo) -> uma linha por (study_id, alvo) valido, com fold."""
    train = pd.read_csv(PROCESSED / "sample_train.csv")
    folds = pd.read_csv(PROCESSED / "folds.csv")[["study_id", "fold"]]
    longo = train.melt(id_vars="study_id", var_name="target", value_name="y_true")
    longo = longo.dropna(subset=["y_true"])
    longo = longo.merge(folds, on="study_id", how="inner", validate="many_to_one")
    assert longo["study_id"].nunique() == 500, "Esperados 500 estudos da amostra."
    return longo


def adicionar_target(df):
    df = df.copy()
    df["target"] = [target_name(c, l) for c, l in zip(df["condition"], df["level"])]
    return df


def metricas_alvo(y_true, y_pred, proba):
    """Metricas de um par (fold, alvo). proba: array (n, 3) na ordem LABELS."""
    proba = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_idx = np.array([LABEL_TO_INT[y] for y in y_true])
    pesos = np.array([CLASS_WEIGHTS_OFICIAIS[y] for y in y_true])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": recall_score(y_true, y_pred, labels=LABELS,
                                          average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=LABELS,
                             average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=LABELS,
                                average="weighted", zero_division=0),
        "log_loss": log_loss(y_idx, proba, labels=[0, 1, 2]),
        "log_loss_ponderada": log_loss(y_idx, proba, labels=[0, 1, 2],
                                       sample_weight=pesos),
    }


METRICAS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
            "log_loss", "log_loss_ponderada"]
PROBA_COLS = ["prob_normal_mild", "prob_moderate", "prob_severe"]


def avaliar_oof(oof):
    """
    oof: DataFrame com colunas fold, target, y_true, y_pred, prob_*.
    Retorna (por_fold_alvo, por_fold, resumo) no mesmo formato do baseline.
    """
    linhas = []
    for (fold, alvo), g in oof.groupby(["fold", "target"]):
        m = metricas_alvo(g["y_true"].to_numpy(), g["y_pred"].to_numpy(),
                          g[PROBA_COLS].to_numpy())
        m.update({"fold": fold, "target": alvo, "n": len(g)})
        linhas.append(m)
    por_fold_alvo = pd.DataFrame(linhas)
    por_fold = por_fold_alvo.groupby("fold")[METRICAS].mean().reset_index()
    resumo = pd.DataFrame({
        "metric": METRICAS,
        "mean": [por_fold[m].mean() for m in METRICAS],
        "std": [por_fold[m].std(ddof=1) for m in METRICAS],
    })
    return por_fold_alvo, por_fold, resumo


def carregar_baseline_oof():
    """Previsoes OOF do baseline trivial (Luis), no formato padrao."""
    b = pd.read_csv(PROCESSED / "baseline_oof_predictions.csv")
    return b[["study_id", "fold", "target", "y_true", "y_pred"] + PROBA_COLS]


def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"
