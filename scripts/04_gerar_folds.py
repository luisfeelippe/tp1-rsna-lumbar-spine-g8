from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

PROCESSED.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5

LABEL_ORDER = ["Normal/Mild", "Moderate", "Severe"]

print("=" * 72)
print("TP1 RSNA G8 - MODULO 06")
print("GERACAO DOS FOLDS DE VALIDACAO CRUZADA")
print("=" * 72)

# ---------------------------------------------------------------------
# 1. CARREGAMENTO
# ---------------------------------------------------------------------

sample_studies = pd.read_csv(
    PROCESSED / "sample_studies.csv"
)

sample_train = pd.read_csv(
    PROCESSED / "sample_train.csv"
)

target_cols = [
    c for c in sample_train.columns
    if c != "study_id"
]

print("\n===== 1. CARREGAMENTO =====")

print(f"Estudos em sample_studies: {len(sample_studies)}")
print(f"Estudos em sample_train: {len(sample_train)}")
print(f"Alvos: {len(target_cols)}")

if len(sample_studies) != 500:
    raise RuntimeError(
        f"Esperado 500 estudos, encontrado {len(sample_studies)}."
    )

if sample_studies["study_id"].duplicated().any():
    raise RuntimeError("study_id duplicado em sample_studies.csv")

if sample_train["study_id"].duplicated().any():
    raise RuntimeError("study_id duplicado em sample_train.csv")

if set(sample_studies["study_id"]) != set(sample_train["study_id"]):
    raise RuntimeError(
        "sample_studies e sample_train possuem study_ids diferentes."
    )

print("OK: manifestos consistentes.")

# ---------------------------------------------------------------------
# 2. CHAVE DE ESTRATIFICACAO
# ---------------------------------------------------------------------

print("\n===== 2. CHAVE DE ESTRATIFICACAO =====")

manifest = sample_studies.copy()

manifest["mandatory_rare_severe"] = (
    manifest["mandatory_rare_severe"]
    .astype(str)
    .str.lower()
    .map({"true": True, "false": False})
)

if manifest["mandatory_rare_severe"].isna().any():
    raise RuntimeError(
        "Falha ao interpretar mandatory_rare_severe."
    )

def make_key(row):
    if (
        row["study_stratum"] == "Severe"
        and row["mandatory_rare_severe"]
    ):
        return "Severe_rare_mandatory"

    return row["study_stratum"]


manifest["stratification_key"] = manifest.apply(
    make_key,
    axis=1,
)

key_counts = (
    manifest["stratification_key"]
    .value_counts()
    .sort_index()
)

print(key_counts.to_string())

if (key_counts < N_SPLITS).any():
    raise RuntimeError(
        "Existe estrato com menos casos que o numero de folds."
    )

# ---------------------------------------------------------------------
# 3. STRATIFIED 5-FOLD
# ---------------------------------------------------------------------

print("\n===== 3. GERANDO 5 FOLDS =====")

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

manifest["fold"] = -1

for fold, (_, validation_idx) in enumerate(
    skf.split(
        manifest,
        manifest["stratification_key"],
    )
):
    manifest.loc[
        manifest.index[validation_idx],
        "fold",
    ] = fold

manifest["fold"] = manifest["fold"].astype(int)

if (manifest["fold"] < 0).any():
    raise RuntimeError("Existem estudos sem fold.")

print("Folds criados.")

# ---------------------------------------------------------------------
# 4. VALIDACOES BASICAS
# ---------------------------------------------------------------------

print("\n===== 4. VALIDACOES BASICAS =====")

if manifest["study_id"].duplicated().any():
    raise RuntimeError("study_id apareceu mais de uma vez.")

if len(manifest) != 500:
    raise RuntimeError("Quantidade inesperada de estudos.")

fold_sizes = (
    manifest["fold"]
    .value_counts()
    .sort_index()
)

print("Tamanho dos folds:")
print(fold_sizes.to_string())

if set(fold_sizes.index) != set(range(N_SPLITS)):
    raise RuntimeError("Nem todos os folds 0-4 existem.")

if fold_sizes.sum() != 500:
    raise RuntimeError("Os folds nao totalizam 500 estudos.")

print("OK: cada estudo pertence a exatamente um fold.")
print("OK: os 500 estudos estao cobertos.")

# ---------------------------------------------------------------------
# 5. DISTRIBUICAO DOS ESTRATOS POR FOLD
# ---------------------------------------------------------------------

print("\n===== 5. ESTRATOS POR FOLD =====")

fold_strata = (
    manifest
    .groupby(
        ["fold", "stratification_key"],
        observed=True,
    )
    .size()
    .reset_index(name="studies")
)

print(
    fold_strata
    .pivot(
        index="fold",
        columns="stratification_key",
        values="studies",
    )
    .fillna(0)
    .astype(int)
    .to_string()
)

fold_strata.to_csv(
    TABLES / "fold_strata_distribution.csv",
    index=False,
)

# ---------------------------------------------------------------------
# 6. JUNTAR ROTULOS AOS FOLDS
# ---------------------------------------------------------------------

print("\n===== 6. VALIDANDO DISTRIBUICAO DOS ROTULOS =====")

fold_labels = sample_train.merge(
    manifest[
        [
            "study_id",
            "fold",
        ]
    ],
    on="study_id",
    how="inner",
    validate="one_to_one",
)

if len(fold_labels) != 500:
    raise RuntimeError(
        "Merge entre rotulos e folds nao resultou em 500 estudos."
    )

rows = []

for fold in range(N_SPLITS):

    df_fold = fold_labels[
        fold_labels["fold"] == fold
    ]

    for target in target_cols:

        valid_target = df_fold[target].dropna()

        for label in LABEL_ORDER:

            count = int(
                (valid_target == label).sum()
            )

            percentage = (
                count / len(valid_target) * 100
                if len(valid_target) > 0
                else 0.0
            )

            rows.append(
                {
                    "fold": fold,
                    "target": target,
                    "severity": label,
                    "count": count,
                    "percentage": percentage,
                    "valid_labels_in_target": int(
                        len(valid_target)
                    ),
                }
            )

fold_class_distribution = pd.DataFrame(rows)

fold_class_distribution.to_csv(
    TABLES / "fold_class_distribution.csv",
    index=False,
)

# ---------------------------------------------------------------------
# 7. COBERTURA ALVO X CLASSE ENTRE FOLDS
# ---------------------------------------------------------------------

print("\n===== 7. COBERTURA ALVO X CLASSE =====")

coverage = (
    fold_class_distribution
    .groupby(
        ["target", "severity"],
        observed=True,
    )
    .agg(
        sample_count=("count", "sum"),
        folds_present=(
            "count",
            lambda x: int((x > 0).sum()),
        ),
        min_count_per_fold=("count", "min"),
        max_count_per_fold=("count", "max"),
    )
    .reset_index()
)

coverage.to_csv(
    TABLES / "fold_class_coverage.csv",
    index=False,
)

limited_coverage = coverage[
    coverage["folds_present"] < N_SPLITS
].copy()

print(
    f"Combinacoes alvo x classe presentes nos 5 folds: "
    f"{len(coverage) - len(limited_coverage)}/{len(coverage)}"
)

print(
    f"Combinacoes que nao aparecem em todos os folds: "
    f"{len(limited_coverage)}"
)

if len(limited_coverage) > 0:
    print("\nIsso e esperado para classes extremamente raras:")
    print(
        limited_coverage
        .sort_values(
            ["folds_present", "sample_count", "target"]
        )
        .to_string(index=False)
    )

# ---------------------------------------------------------------------
# 8. AUSENCIA DE ROTULOS POR FOLD
# ---------------------------------------------------------------------

print("\n===== 8. ROTULOS AUSENTES POR FOLD =====")

missing_rows = []

for fold in range(N_SPLITS):

    df_fold = fold_labels[
        fold_labels["fold"] == fold
    ]

    missing = int(
        df_fold[target_cols]
        .isna()
        .sum()
        .sum()
    )

    studies_with_missing = int(
        df_fold[target_cols]
        .isna()
        .any(axis=1)
        .sum()
    )

    missing_rows.append(
        {
            "fold": fold,
            "studies": len(df_fold),
            "studies_with_missing_labels": studies_with_missing,
            "missing_labels": missing,
        }
    )

missing_by_fold = pd.DataFrame(missing_rows)

print(missing_by_fold.to_string(index=False))

missing_by_fold.to_csv(
    TABLES / "fold_missing_labels.csv",
    index=False,
)

# ---------------------------------------------------------------------
# 9. EXPORTAR FOLDS
# ---------------------------------------------------------------------

print("\n===== 9. EXPORTANDO FOLDS =====")

folds = manifest[
    [
        "study_id",
        "fold",
        "study_stratum",
        "mandatory_rare_severe",
        "stratification_key",
    ]
].sort_values(
    ["fold", "study_id"]
).reset_index(drop=True)

folds.to_csv(
    PROCESSED / "folds.csv",
    index=False,
)

metadata = {
    "seed": SEED,
    "n_splits": N_SPLITS,
    "sample_size": int(len(manifest)),
    "unit_of_split": "study_id",
    "splitter": "StratifiedKFold",
    "shuffle": True,
    "stratification_key": (
        "study_stratum, com casos obrigatorios de Severe raro "
        "separados em Severe_rare_mandatory"
    ),
    "important_rule": (
        "Todas as imagens/series de um study_id devem herdar "
        "o mesmo fold."
    ),
}

with open(
    PROCESSED / "fold_metadata.json",
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
print(" - data/processed/folds.csv")
print(" - data/processed/fold_metadata.json")
print(" - outputs/tables/fold_strata_distribution.csv")
print(" - outputs/tables/fold_class_distribution.csv")
print(" - outputs/tables/fold_class_coverage.csv")
print(" - outputs/tables/fold_missing_labels.csv")

print("\n===== MODULO 06 CONCLUIDO =====")
