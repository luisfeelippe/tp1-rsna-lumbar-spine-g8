from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    f1_score,
    log_loss,
)

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

TABLES.mkdir(parents=True, exist_ok=True)

LABELS = ["Normal/Mild", "Moderate", "Severe"]
N_SPLITS = 5

print("=" * 72)
print("TP1 RSNA G8 - MODULO 07")
print("BASELINE TRIVIAL - CLASSE MAJORITARIA")
print("=" * 72)

# ---------------------------------------------------------------------
# 1. CARREGAMENTO
# ---------------------------------------------------------------------

print("\n===== 1. CARREGAMENTO =====")

train = pd.read_csv(
    PROCESSED / "sample_train.csv"
)

folds = pd.read_csv(
    PROCESSED / "folds.csv"
)

target_cols = [
    col for col in train.columns
    if col != "study_id"
]

data = train.merge(
    folds[["study_id", "fold"]],
    on="study_id",
    how="inner",
    validate="one_to_one",
)

if len(data) != 500:
    raise RuntimeError(
        f"Esperados 500 estudos apos merge, encontrados {len(data)}."
    )

if data["study_id"].duplicated().any():
    raise RuntimeError("study_id duplicado.")

if set(data["fold"]) != set(range(N_SPLITS)):
    raise RuntimeError("Folds 0-4 nao encontrados.")

print(f"Estudos: {len(data)}")
print(f"Alvos: {len(target_cols)}")
print(f"Folds: {sorted(data['fold'].unique())}")
print("OK: dados e folds consistentes.")

# ---------------------------------------------------------------------
# FUNCOES
# ---------------------------------------------------------------------

def safe_div(num, den):
    if den == 0:
        return np.nan
    return num / den


def majority_with_tiebreak(series):
    counts = (
        series
        .dropna()
        .value_counts()
        .reindex(LABELS, fill_value=0)
    )

    maximum = counts.max()

    tied = [
        label
        for label in LABELS
        if counts[label] == maximum
    ]

    # Desempate deterministico seguindo LABELS.
    return tied[0], counts


def class_metrics(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    total = cm.sum()

    rows = []

    for i, label in enumerate(LABELS):

        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)

        sensitivity = safe_div(
            tp,
            tp + fn,
        )

        specificity = safe_div(
            tn,
            tn + fp,
        )

        rows.append(
            {
                "severity": label,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "sensitivity": sensitivity,
                "specificity": specificity,
            }
        )

    return rows, cm


# ---------------------------------------------------------------------
# 2. EXECUTAR OS 5 FOLDS
# ---------------------------------------------------------------------

print("\n===== 2. EXECUTANDO VALIDACAO CRUZADA =====")

metric_rows = []
class_metric_rows = []
prior_rows = []
prediction_rows = []
confusion_rows = []

for fold in range(N_SPLITS):

    print(f"\n--- FOLD {fold} ---")

    train_fold = data[
        data["fold"] != fold
    ].copy()

    val_fold = data[
        data["fold"] == fold
    ].copy()

    print(
        f"Treino: {len(train_fold)} | "
        f"Validacao: {len(val_fold)}"
    )

    for target in target_cols:

        y_train = (
            train_fold[target]
            .dropna()
            .astype(str)
        )

        val_target = (
            val_fold[
                ["study_id", target]
            ]
            .dropna(subset=[target])
            .copy()
        )

        y_true = (
            val_target[target]
            .astype(str)
            .to_numpy()
        )

        if len(y_train) == 0:
            raise RuntimeError(
                f"Sem rotulos de treino: fold={fold}, target={target}"
            )

        if len(y_true) == 0:
            print(
                f"ATENCAO: sem validacao para {target} no fold {fold}"
            )
            continue

        majority_class, train_counts = majority_with_tiebreak(
            y_train
        )

        y_pred = np.repeat(
            majority_class,
            len(y_true),
        )

        # Probabilidades triviais = prevalencia das classes
        # calculada SOMENTE no treino da dobra.
        total_train = int(train_counts.sum())

        train_priors = (
            train_counts / total_train
        ).astype(float)

        # Segurança numerica para log-loss.
        eps = 1e-15

        probabilities = np.array(
            [
                [
                    max(
                        float(train_priors[label]),
                        eps,
                    )
                    for label in LABELS
                ]
                for _ in range(len(y_true))
            ],
            dtype=float,
        )

        probabilities = (
            probabilities
            / probabilities.sum(axis=1, keepdims=True)
        )

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        # Balanced Accuracy multiclasses:
        # macro recall sobre as 3 classes fixas.
        # Assim, classes ausentes em uma dobra nao alteram
        # artificialmente o numero de classes consideradas.
        balanced_accuracy = recall_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        f1_macro = f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        f1_weighted = f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        )

        # O sklearn exige ordem lexicografica para as classes
        # utilizadas no log_loss. As probabilidades foram criadas
        # na ordem LABELS, entao reordenamos somente para esta metrica.
        log_loss_labels = sorted(LABELS)

        log_loss_indices = [
            LABELS.index(label)
            for label in log_loss_labels
        ]

        log_loss_probabilities = probabilities[
            :,
            log_loss_indices,
        ]

        ll = log_loss(
            y_true,
            log_loss_probabilities,
            labels=log_loss_labels,
        )

        metric_rows.append(
            {
                "fold": fold,
                "target": target,
                "validation_samples": len(y_true),
                "majority_class": majority_class,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "log_loss": ll,
            }
        )

        for label in LABELS:
            prior_rows.append(
                {
                    "fold": fold,
                    "target": target,
                    "severity": label,
                    "train_count": int(
                        train_counts[label]
                    ),
                    "train_probability": float(
                        train_priors[label]
                    ),
                }
            )

        per_class, cm = class_metrics(
            y_true,
            y_pred,
        )

        for row in per_class:
            row.update(
                {
                    "fold": fold,
                    "target": target,
                }
            )

            class_metric_rows.append(row)

        for true_idx, true_label in enumerate(LABELS):
            for pred_idx, pred_label in enumerate(LABELS):

                confusion_rows.append(
                    {
                        "fold": fold,
                        "target": target,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": int(
                            cm[true_idx, pred_idx]
                        ),
                    }
                )

        for i, (_, row) in enumerate(
            val_target.iterrows()
        ):

            prediction_rows.append(
                {
                    "study_id": int(row["study_id"]),
                    "fold": fold,
                    "target": target,
                    "y_true": str(row[target]),
                    "y_pred": majority_class,
                    "prob_normal_mild": probabilities[i, 0],
                    "prob_moderate": probabilities[i, 1],
                    "prob_severe": probabilities[i, 2],
                }
            )

print("\nOK: validacao cruzada concluida.")

# ---------------------------------------------------------------------
# 3. DATAFRAMES
# ---------------------------------------------------------------------

metrics = pd.DataFrame(metric_rows)
class_metrics_df = pd.DataFrame(class_metric_rows)
priors = pd.DataFrame(prior_rows)
predictions = pd.DataFrame(prediction_rows)
confusions = pd.DataFrame(confusion_rows)

print("\n===== 3. QUANTIDADE DE RESULTADOS =====")

print(f"Metricas fold x alvo: {len(metrics)}")
print(f"Metricas por classe: {len(class_metrics_df)}")
print(f"Previsoes OOF: {len(predictions)}")

expected_metric_rows = N_SPLITS * len(target_cols)

if len(metrics) != expected_metric_rows:
    print(
        "ATENCAO: nem todos os fold x alvo produziram metricas."
    )

# ---------------------------------------------------------------------
# 4. RESUMO POR FOLD
# ---------------------------------------------------------------------

print("\n===== 4. RESUMO POR FOLD =====")

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "log_loss",
]

fold_summary = (
    metrics
    .groupby("fold")[metric_columns]
    .mean()
    .reset_index()
)

print(fold_summary.to_string(index=False))

# ---------------------------------------------------------------------
# 5. MEDIA E DESVIO ENTRE FOLDS
# ---------------------------------------------------------------------

print("\n===== 5. MEDIA +/- DESVIO ENTRE FOLDS =====")

overall_rows = []

for metric in metric_columns:

    values = fold_summary[metric]

    mean = float(values.mean())
    std = float(values.std(ddof=1))

    overall_rows.append(
        {
            "metric": metric,
            "mean": mean,
            "std": std,
            "mean_plus_minus_std": (
                f"{mean:.4f} ± {std:.4f}"
            ),
        }
    )

overall_summary = pd.DataFrame(overall_rows)

print(overall_summary.to_string(index=False))

# ---------------------------------------------------------------------
# 6. RESUMO POR ALVO
# ---------------------------------------------------------------------

print("\n===== 6. RESUMO POR ALVO =====")

target_summary = (
    metrics
    .groupby("target")[metric_columns]
    .agg(["mean", "std"])
)

target_summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in target_summary.columns
]

target_summary = (
    target_summary
    .reset_index()
)

print(
    target_summary[
        [
            "target",
            "balanced_accuracy_mean",
            "f1_macro_mean",
            "log_loss_mean",
        ]
    ]
    .sort_values("f1_macro_mean")
    .to_string(index=False)
)

# ---------------------------------------------------------------------
# 7. RESUMO POR CLASSE
# ---------------------------------------------------------------------

print("\n===== 7. SENSIBILIDADE / ESPECIFICIDADE =====")

class_summary = (
    class_metrics_df
    .groupby("severity")[
        ["sensitivity", "specificity"]
    ]
    .agg(["mean", "std"])
)

class_summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in class_summary.columns
]

class_summary = (
    class_summary
    .reindex(LABELS)
    .reset_index()
)

print(class_summary.to_string(index=False))

# ---------------------------------------------------------------------
# 8. VALIDACOES OOF
# ---------------------------------------------------------------------

print("\n===== 8. VALIDACAO DAS PREVISOES OOF =====")

duplicates = predictions.duplicated(
    subset=[
        "study_id",
        "target",
    ]
).sum()

print(
    f"Pares study_id/target duplicados nas previsoes OOF: "
    f"{duplicates}"
)

if duplicates != 0:
    raise RuntimeError(
        "Ha previsoes OOF duplicadas."
    )

expected_predictions = int(
    data[target_cols]
    .notna()
    .sum()
    .sum()
)

print(
    f"Rotulos validos esperados: {expected_predictions}"
)

print(
    f"Previsoes OOF geradas: {len(predictions)}"
)

if len(predictions) != expected_predictions:
    raise RuntimeError(
        "Quantidade de previsoes OOF difere dos rotulos validos."
    )

print("OK: cada rotulo valido recebeu exatamente uma previsao OOF.")

# ---------------------------------------------------------------------
# 9. EXPORTAR
# ---------------------------------------------------------------------

print("\n===== 9. EXPORTANDO =====")

metrics.to_csv(
    TABLES / "baseline_metrics_by_fold_target.csv",
    index=False,
)

fold_summary.to_csv(
    TABLES / "baseline_metrics_by_fold.csv",
    index=False,
)

overall_summary.to_csv(
    TABLES / "baseline_metrics_summary.csv",
    index=False,
)

target_summary.to_csv(
    TABLES / "baseline_metrics_by_target.csv",
    index=False,
)

class_metrics_df.to_csv(
    TABLES / "baseline_class_metrics_raw.csv",
    index=False,
)

class_summary.to_csv(
    TABLES / "baseline_class_metrics_summary.csv",
    index=False,
)

priors.to_csv(
    TABLES / "baseline_train_priors.csv",
    index=False,
)

confusions.to_csv(
    TABLES / "baseline_confusion_matrices.csv",
    index=False,
)

predictions.to_csv(
    PROCESSED / "baseline_oof_predictions.csv",
    index=False,
)

metadata = {
    "baseline": "classe majoritaria por alvo",
    "probabilistic_baseline": (
        "prevalencia das classes no conjunto de treino da dobra"
    ),
    "fit_scope": (
        "classe majoritaria e probabilidades calculadas "
        "somente nos 4 folds de treino"
    ),
    "validation": "5-fold cross-validation",
    "split_unit": "study_id",
    "labels": LABELS,
    "metrics": metric_columns,
    "balanced_accuracy_definition": (
        "macro recall calculado sobre as tres classes fixas: "
        "Normal/Mild, Moderate e Severe"
    ),
    "log_loss_class_order": sorted(LABELS),
    "missing_labels": (
        "ignorados no treino e na avaliacao do alvo correspondente"
    ),
}

with open(
    PROCESSED / "baseline_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("Arquivos gerados:")

for path in [
    TABLES / "baseline_metrics_by_fold_target.csv",
    TABLES / "baseline_metrics_by_fold.csv",
    TABLES / "baseline_metrics_summary.csv",
    TABLES / "baseline_metrics_by_target.csv",
    TABLES / "baseline_class_metrics_raw.csv",
    TABLES / "baseline_class_metrics_summary.csv",
    TABLES / "baseline_train_priors.csv",
    TABLES / "baseline_confusion_matrices.csv",
    PROCESSED / "baseline_oof_predictions.csv",
    PROCESSED / "baseline_metadata.json",
]:
    print(
        f" - {path.relative_to(ROOT)} | "
        f"{path.stat().st_size} bytes"
    )

print("\n===== MODULO 07 CONCLUIDO =====")
