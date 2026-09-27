from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "tables"

OUT.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = RAW / "train.csv"
SERIES_PATH = RAW / "train_series_descriptions.csv"
COORD_PATH = RAW / "train_label_coordinates.csv"

EXPECTED_FILES = [
    TRAIN_PATH,
    SERIES_PATH,
    COORD_PATH,
]

EXPECTED_LABELS = {"Normal/Mild", "Moderate", "Severe"}

print("=" * 72)
print("TP1 RSNA G8 - MODULO 03")
print("INSPECAO E VALIDACAO DOS DADOS")
print("=" * 72)

print("\n===== 1. VERIFICACAO DOS ARQUIVOS =====")

missing_files = [str(p) for p in EXPECTED_FILES if not p.exists()]

if missing_files:
    print("ERRO: arquivos obrigatorios ausentes:")
    for p in missing_files:
        print(" -", p)
    sys.exit(1)

for path in EXPECTED_FILES:
    print(f"OK: {path.name} | {path.stat().st_size} bytes")

print("\n===== 2. CARREGAMENTO =====")

train = pd.read_csv(TRAIN_PATH)
series = pd.read_csv(SERIES_PATH)
coords = pd.read_csv(COORD_PATH)

print(f"train.csv: {train.shape[0]} linhas x {train.shape[1]} colunas")
print(f"series: {series.shape[0]} linhas x {series.shape[1]} colunas")
print(f"coordinates: {coords.shape[0]} linhas x {coords.shape[1]} colunas")

target_cols = [c for c in train.columns if c != "study_id"]

print("\n===== 3. ESTRUTURA DOS ROTULOS =====")
print(f"Quantidade de estudos: {train['study_id'].nunique()}")
print(f"Quantidade de alvos: {len(target_cols)}")

levels_from_targets = sorted(
    {
        "_".join(col.split("_")[-2:]).upper().replace("_", "/")
        for col in target_cols
    }
)

conditions_from_targets = sorted(
    {
        "_".join(col.split("_")[:-2])
        for col in target_cols
    }
)

print("\nCondicoes codificadas nas colunas:")
for item in conditions_from_targets:
    print(" -", item)

print("\nNiveis codificados nas colunas:")
for item in levels_from_targets:
    print(" -", item)

print("\n===== 4. CLASSES EXISTENTES =====")

observed_labels = set(
    train[target_cols]
    .stack()
    .dropna()
    .astype(str)
    .unique()
)

print("Classes encontradas:", sorted(observed_labels))

unexpected_labels = observed_labels - EXPECTED_LABELS

if unexpected_labels:
    print("ATENCAO - classes inesperadas:", sorted(unexpected_labels))
else:
    print("OK: somente as tres classes esperadas foram encontradas.")

print("\n===== 5. VALORES AUSENTES =====")

missing_by_target = (
    train[target_cols]
    .isna()
    .sum()
    .rename("missing_count")
    .to_frame()
)

missing_by_target["total"] = len(train)
missing_by_target["missing_pct"] = (
    missing_by_target["missing_count"] / missing_by_target["total"] * 100
)

missing_by_target.index.name = "target"
missing_by_target = missing_by_target.reset_index()

print(f"Total de valores ausentes nos 25 alvos: "
      f"{int(missing_by_target['missing_count'].sum())}")

print("\nTop 10 alvos com mais valores ausentes:")
print(
    missing_by_target
    .sort_values(["missing_count", "target"], ascending=[False, True])
    .head(10)
    .to_string(index=False)
)

print("\n===== 6. DISTRIBUICAO GLOBAL DAS CLASSES =====")

long_labels = (
    train
    .melt(
        id_vars="study_id",
        value_vars=target_cols,
        var_name="target",
        value_name="severity",
    )
)

global_distribution = (
    long_labels["severity"]
    .value_counts(dropna=False)
    .rename_axis("severity")
    .reset_index(name="count")
)

global_distribution["percentage"] = (
    global_distribution["count"] / len(long_labels) * 100
)

print(global_distribution.to_string(index=False))

print("\n===== 7. DISTRIBUICAO POR ALVO =====")

target_distribution = (
    long_labels
    .dropna(subset=["severity"])
    .groupby(["target", "severity"], observed=True)
    .size()
    .reset_index(name="count")
)

target_totals = (
    target_distribution
    .groupby("target")["count"]
    .transform("sum")
)

target_distribution["percentage"] = (
    target_distribution["count"] / target_totals * 100
)

print(f"Linhas na tabela alvo x classe: {len(target_distribution)}")

print("\n===== 8. SERIES DE RESSONANCIA =====")

series_distribution = (
    series["series_description"]
    .value_counts(dropna=False)
    .rename_axis("series_description")
    .reset_index(name="count")
)

series_distribution["percentage"] = (
    series_distribution["count"] / len(series) * 100
)

print(series_distribution.to_string(index=False))

series_per_study = (
    series
    .groupby("study_id")
    .size()
    .rename("series_count")
)

print("\nSeries por estudo:")
print(series_per_study.describe().to_string())

print("\n===== 9. COORDENADAS / ANOTACOES =====")

condition_distribution = (
    coords["condition"]
    .value_counts(dropna=False)
    .rename_axis("condition")
    .reset_index(name="count")
)

level_distribution = (
    coords["level"]
    .value_counts(dropna=False)
    .rename_axis("level")
    .reset_index(name="count")
)

print("\nCondicoes:")
print(condition_distribution.to_string(index=False))

print("\nNiveis:")
print(level_distribution.to_string(index=False))

print("\n===== 10. DUPLICATAS =====")

train_dup_study = int(train["study_id"].duplicated().sum())

series_dup = int(
    series.duplicated(
        subset=["study_id", "series_id"]
    ).sum()
)

coords_dup = int(coords.duplicated().sum())

print(f"study_id duplicado em train.csv: {train_dup_study}")
print(f"pares study_id/series_id duplicados em series: {series_dup}")
print(f"linhas totalmente duplicadas em coordinates: {coords_dup}")

print("\n===== 11. CONSISTENCIA ENTRE OS ARQUIVOS =====")

train_ids = set(train["study_id"])
series_ids = set(series["study_id"])
coords_ids = set(coords["study_id"])

checks = [
    {
        "check": "studies_train",
        "value": len(train_ids),
    },
    {
        "check": "studies_series",
        "value": len(series_ids),
    },
    {
        "check": "studies_coordinates",
        "value": len(coords_ids),
    },
    {
        "check": "train_not_in_series",
        "value": len(train_ids - series_ids),
    },
    {
        "check": "train_not_in_coordinates",
        "value": len(train_ids - coords_ids),
    },
    {
        "check": "series_not_in_train",
        "value": len(series_ids - train_ids),
    },
    {
        "check": "coordinates_not_in_train",
        "value": len(coords_ids - train_ids),
    },
]

checks_df = pd.DataFrame(checks)

print(checks_df.to_string(index=False))

print("\n===== 12. RESUMO DO DATASET =====")

summary = pd.DataFrame(
    [
        ["studies", train["study_id"].nunique()],
        ["targets", len(target_cols)],
        ["series_rows", len(series)],
        ["unique_series", series["series_id"].nunique()],
        ["coordinate_rows", len(coords)],
        ["total_possible_labels", len(train) * len(target_cols)],
        ["missing_labels", int(train[target_cols].isna().sum().sum())],
        ["unique_series_descriptions", series["series_description"].nunique()],
        ["unique_coordinate_conditions", coords["condition"].nunique()],
        ["unique_coordinate_levels", coords["level"].nunique()],
    ],
    columns=["metric", "value"],
)

print(summary.to_string(index=False))

print("\n===== 13. SALVANDO ARTEFATOS =====")

summary.to_csv(
    OUT / "dataset_summary.csv",
    index=False,
)

missing_by_target.to_csv(
    OUT / "missing_by_target.csv",
    index=False,
)

global_distribution.to_csv(
    OUT / "global_class_distribution.csv",
    index=False,
)

target_distribution.to_csv(
    OUT / "class_distribution_by_target.csv",
    index=False,
)

series_distribution.to_csv(
    OUT / "series_description_distribution.csv",
    index=False,
)

condition_distribution.to_csv(
    OUT / "coordinate_condition_distribution.csv",
    index=False,
)

level_distribution.to_csv(
    OUT / "coordinate_level_distribution.csv",
    index=False,
)

checks_df.to_csv(
    OUT / "integrity_checks.csv",
    index=False,
)

series_per_study.rename_axis("study_id").reset_index().to_csv(
    OUT / "series_per_study.csv",
    index=False,
)

print("Arquivos gerados:")

for path in sorted(OUT.glob("*.csv")):
    print(f" - {path.name} | {path.stat().st_size} bytes")

print("\n===== MODULO 03 CONCLUIDO =====")
