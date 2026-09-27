from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

errors = []

print("=" * 72)
print("TP1 RSNA G8 - MODULO 08")
print("AUDITORIA FINAL DA CONTRIBUICAO DE LUIS FELIPE")
print("=" * 72)

# ------------------------------------------------------------------
# 1. Arquivos obrigatorios
# ------------------------------------------------------------------

required = [
    ROOT / "scripts/01_inspecionar_dados.py",
    ROOT / "scripts/02_eda.py",
    ROOT / "scripts/03_gerar_amostra.py",
    ROOT / "scripts/04_gerar_folds.py",
    ROOT / "scripts/05_baseline_trivial.py",

    PROCESSED / "sample_studies.csv",
    PROCESSED / "sample_train.csv",
    PROCESSED / "sample_series_descriptions.csv",
    PROCESSED / "sample_label_coordinates.csv",
    PROCESSED / "sample_metadata.json",

    PROCESSED / "folds.csv",
    PROCESSED / "fold_metadata.json",

    PROCESSED / "baseline_oof_predictions.csv",
    PROCESSED / "baseline_metadata.json",

    TABLES / "baseline_metrics_summary.csv",
    TABLES / "baseline_metrics_by_fold.csv",
]

print("\n===== 1. ARQUIVOS OBRIGATORIOS =====")

for path in required:
    if path.exists():
        print(f"OK: {path.relative_to(ROOT)}")
    else:
        errors.append(f"Arquivo ausente: {path}")
        print(f"ERRO: {path.relative_to(ROOT)}")

# ------------------------------------------------------------------
# 2. Amostra
# ------------------------------------------------------------------

print("\n===== 2. AMOSTRA =====")

sample = pd.read_csv(PROCESSED / "sample_studies.csv")
sample_train = pd.read_csv(PROCESSED / "sample_train.csv")

print(f"sample_studies: {len(sample)}")
print(f"sample_train: {len(sample_train)}")
print(f"study_id unicos: {sample['study_id'].nunique()}")

if len(sample) != 500:
    errors.append("A amostra nao possui exatamente 500 estudos.")

if sample["study_id"].nunique() != 500:
    errors.append("Existem study_ids duplicados na amostra.")

if set(sample["study_id"]) != set(sample_train["study_id"]):
    errors.append("sample_studies e sample_train nao possuem os mesmos IDs.")

# ------------------------------------------------------------------
# 3. Folds
# ------------------------------------------------------------------

print("\n===== 3. FOLDS =====")

folds = pd.read_csv(PROCESSED / "folds.csv")

fold_counts = (
    folds["fold"]
    .value_counts()
    .sort_index()
)

print(fold_counts.to_string())

if len(folds) != 500:
    errors.append("folds.csv nao possui 500 estudos.")

if folds["study_id"].nunique() != 500:
    errors.append("Existem study_ids repetidos em folds.csv.")

if set(fold_counts.index) != {0, 1, 2, 3, 4}:
    errors.append("Os folds 0-4 nao estao todos presentes.")

if not all(fold_counts.get(i, 0) == 100 for i in range(5)):
    errors.append("Os folds nao possuem exatamente 100 estudos cada.")

if set(folds["study_id"]) != set(sample["study_id"]):
    errors.append("folds.csv nao cobre exatamente a amostra.")

# ------------------------------------------------------------------
# 4. Baseline
# ------------------------------------------------------------------

print("\n===== 4. BASELINE =====")

metrics_fold_target = pd.read_csv(
    TABLES / "baseline_metrics_by_fold_target.csv"
)

summary = pd.read_csv(
    TABLES / "baseline_metrics_summary.csv"
)

predictions = pd.read_csv(
    PROCESSED / "baseline_oof_predictions.csv"
)

print(f"fold x alvo: {len(metrics_fold_target)}")
print(f"previsoes OOF: {len(predictions)}")

if len(metrics_fold_target) != 125:
    errors.append(
        "Esperadas 125 combinacoes fold x alvo no baseline."
    )

if len(predictions) != 12387:
    errors.append(
        "Esperadas 12387 previsoes OOF."
    )

duplicates = predictions.duplicated(
    subset=["study_id", "target"]
).sum()

print(f"duplicatas study_id/target OOF: {duplicates}")

if duplicates != 0:
    errors.append("Existem previsoes OOF duplicadas.")

print("\nResumo final:")
print(summary.to_string(index=False))

expected_metrics = {
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "log_loss",
}

if set(summary["metric"]) != expected_metrics:
    errors.append(
        "O resumo nao possui exatamente as metricas esperadas."
    )

# ------------------------------------------------------------------
# 5. Figuras
# ------------------------------------------------------------------

print("\n===== 5. FIGURAS =====")

figures = sorted(FIGURES.glob("*.png"))

for fig in figures:
    print(f"{fig.name} | {fig.stat().st_size} bytes")

if len(figures) < 4:
    errors.append("Menos de 4 figuras de EDA encontradas.")

# ------------------------------------------------------------------
# 6. Verificacao final
# ------------------------------------------------------------------

print("\n===== 6. RESULTADO DA AUDITORIA =====")

if errors:
    print("AUDITORIA_FALHOU")

    for error in errors:
        print(" -", error)

    sys.exit(1)

print("TODAS_AS_VALIDACOES_OK")
print("A contribuicao tecnica de Luis esta consistente.")
