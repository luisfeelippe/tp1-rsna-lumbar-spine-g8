from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

PROCESSED.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLE_SIZE = 500

EXPECTED_SERIES = {
    "Sagittal T1",
    "Sagittal T2/STIR",
    "Axial T2",
}

LABEL_ORDER = ["Normal/Mild", "Moderate", "Severe"]

train = pd.read_csv(RAW / "train.csv")
series = pd.read_csv(RAW / "train_series_descriptions.csv")
coords = pd.read_csv(RAW / "train_label_coordinates.csv")

target_cols = [c for c in train.columns if c != "study_id"]

print("=" * 72)
print("TP1 RSNA G8 - MODULO 05")
print("AMOSTRAGEM REPRODUZIVEL")
print("=" * 72)

print(f"\nSEED = {SEED}")
print(f"SAMPLE_SIZE = {SAMPLE_SIZE}")

# ---------------------------------------------------------------------
# 1. ESTUDOS ELEGIVEIS
# ---------------------------------------------------------------------

print("\n===== 1. DEFININDO ESTUDOS ELEGIVEIS =====")

series_sets = (
    series.groupby("study_id")["series_description"]
    .agg(lambda x: set(x))
)

complete_series_ids = set(
    series_sets[
        series_sets.apply(lambda s: EXPECTED_SERIES.issubset(s))
    ].index
)

coord_ids = set(coords["study_id"])

eligible_ids = complete_series_ids & coord_ids

eligible = train[
    train["study_id"].isin(eligible_ids)
].copy()

print(f"Estudos totais: {len(train)}")
print(f"Com as 3 series principais: {len(complete_series_ids)}")
print(f"Com coordenadas: {len(coord_ids)}")
print(f"Elegiveis finais: {len(eligible)}")
print(f"Excluidos: {len(train) - len(eligible)}")

# ---------------------------------------------------------------------
# 2. PERFIL DE SEVERIDADE DO ESTUDO
# ---------------------------------------------------------------------

print("\n===== 2. PERFIL DE SEVERIDADE =====")

def study_stratum(row):
    values = row[target_cols].dropna()

    if (values == "Severe").any():
        return "Severe"

    if (values == "Moderate").any():
        return "Moderate"

    return "Normal/Mild"


eligible["study_stratum"] = eligible.apply(
    study_stratum,
    axis=1,
)

stratum_full = (
    eligible["study_stratum"]
    .value_counts()
    .reindex(LABEL_ORDER, fill_value=0)
)

print(stratum_full.to_string())

# ---------------------------------------------------------------------
# 3. IDENTIFICAR SEVERE RARO
# ---------------------------------------------------------------------

print("\n===== 3. ALVOS SEVERE RAROS =====")

severe_counts = (
    eligible[target_cols]
    .eq("Severe")
    .sum()
    .sort_values()
)

# Critério explícito:
# todo alvo com <= 20 exemplos Severe é considerado extremamente raro.
rare_severe_targets = severe_counts[
    severe_counts <= 20
].index.tolist()

print(f"Quantidade de alvos Severe raros (<=20): {len(rare_severe_targets)}")

for target in rare_severe_targets:
    print(f" - {target}: {int(severe_counts[target])}")

mandatory_mask = (
    eligible[rare_severe_targets]
    .eq("Severe")
    .any(axis=1)
)

mandatory = eligible[mandatory_mask].copy()
remaining = eligible[~mandatory_mask].copy()

mandatory_ids = set(mandatory["study_id"])

print(f"\nEstudos obrigatorios por Severe raro: {len(mandatory)}")

# ---------------------------------------------------------------------
# 4. AMOSTRAGEM ESTRATIFICADA DO RESTANTE
# ---------------------------------------------------------------------

print("\n===== 4. AMOSTRAGEM ESTRATIFICADA =====")

remaining_needed = SAMPLE_SIZE - len(mandatory)

if remaining_needed <= 0:
    raise RuntimeError(
        "Quantidade de estudos obrigatorios excedeu o tamanho da amostra."
    )

# Distribuição proporcional entre os estudos não obrigatórios.
stratum_counts = remaining["study_stratum"].value_counts()

raw_quotas = (
    stratum_counts / len(remaining) * remaining_needed
)

quotas = np.floor(raw_quotas).astype(int)

# Distribui as vagas restantes pelos maiores restos fracionários.
leftover = remaining_needed - int(quotas.sum())

remainders = (
    raw_quotas - quotas
).sort_values(ascending=False)

for stratum in remainders.index[:leftover]:
    quotas[stratum] += 1

print("Quotas para sorteio:")

for stratum in LABEL_ORDER:
    print(f" - {stratum}: {int(quotas.get(stratum, 0))}")

sampled_parts = []

for stratum in LABEL_ORDER:
    group = remaining[
        remaining["study_stratum"] == stratum
    ]

    n = int(quotas.get(stratum, 0))

    if n > len(group):
        raise RuntimeError(
            f"Quota {n} maior que grupo {stratum} ({len(group)})."
        )

    if n > 0:
        sampled_parts.append(
            group.sample(
                n=n,
                random_state=SEED,
                replace=False,
            )
        )

sampled_random = pd.concat(
    sampled_parts,
    ignore_index=False,
)

sample = pd.concat(
    [mandatory, sampled_random],
    ignore_index=False,
)

sample = sample.drop_duplicates(subset="study_id").copy()

if len(sample) != SAMPLE_SIZE:
    raise RuntimeError(
        f"Amostra deveria ter {SAMPLE_SIZE}, mas possui {len(sample)}."
    )

# Embaralhamento final determinístico.
sample = sample.sample(
    frac=1,
    random_state=SEED,
).reset_index(drop=True)

sample["mandatory_rare_severe"] = (
    sample["study_id"].isin(mandatory_ids)
)

# ---------------------------------------------------------------------
# 5. VALIDAÇÕES
# ---------------------------------------------------------------------

print("\n===== 5. VALIDACOES =====")

assert sample["study_id"].is_unique
assert set(sample["study_id"]).issubset(eligible_ids)
assert mandatory_ids.issubset(set(sample["study_id"]))

print("OK: study_id unico.")
print("OK: todos os estudos pertencem ao conjunto elegivel.")
print("OK: todos os casos Severe raros foram preservados.")

# ---------------------------------------------------------------------
# 6. COBERTURA DE TODAS AS CLASSES
# ---------------------------------------------------------------------

print("\n===== 6. COBERTURA ALVO X CLASSE =====")

coverage_rows = []
coverage_failures = []

for target in target_cols:
    for label in LABEL_ORDER:

        eligible_count = int(
            (eligible[target] == label).sum()
        )

        sample_count = int(
            (sample[target] == label).sum()
        )

        coverage_rows.append(
            {
                "target": target,
                "severity": label,
                "eligible_count": eligible_count,
                "sample_count": sample_count,
            }
        )

        if eligible_count > 0 and sample_count == 0:
            coverage_failures.append(
                (target, label, eligible_count)
            )

coverage_df = pd.DataFrame(coverage_rows)

if coverage_failures:
    print("ERRO: classes presentes nos elegiveis desapareceram da amostra!")

    for target, label, count in coverage_failures:
        print(f" - {target} | {label} | elegiveis={count}")

    raise RuntimeError(
        "A amostra nao preservou todas as classes existentes."
    )

print("OK: todas as classes existentes nos elegiveis aparecem na amostra.")

# ---------------------------------------------------------------------
# 7. DISTRIBUICAO GLOBAL: FULL VS SAMPLE
# ---------------------------------------------------------------------

print("\n===== 7. COMPARACAO FULL VS SAMPLE =====")

def global_distribution(df):
    melted = (
        df[target_cols]
        .melt(value_name="severity")
        .dropna(subset=["severity"])
    )

    counts = (
        melted["severity"]
        .value_counts()
        .reindex(LABEL_ORDER, fill_value=0)
    )

    pct = counts / counts.sum() * 100

    return counts, pct


full_counts, full_pct = global_distribution(eligible)
sample_counts, sample_pct = global_distribution(sample)

comparison = pd.DataFrame(
    {
        "severity": LABEL_ORDER,
        "eligible_count": [
            int(full_counts[x]) for x in LABEL_ORDER
        ],
        "eligible_pct": [
            float(full_pct[x]) for x in LABEL_ORDER
        ],
        "sample_count": [
            int(sample_counts[x]) for x in LABEL_ORDER
        ],
        "sample_pct": [
            float(sample_pct[x]) for x in LABEL_ORDER
        ],
    }
)

comparison["percentage_point_difference"] = (
    comparison["sample_pct"] - comparison["eligible_pct"]
)

print(comparison.to_string(index=False))

# ---------------------------------------------------------------------
# 8. PERFIL DA AMOSTRA
# ---------------------------------------------------------------------

print("\n===== 8. PERFIL DA AMOSTRA =====")

sample_strata = (
    sample["study_stratum"]
    .value_counts()
    .reindex(LABEL_ORDER, fill_value=0)
    .rename_axis("study_stratum")
    .reset_index(name="studies")
)

sample_strata["percentage"] = (
    sample_strata["studies"] / len(sample) * 100
)

print(sample_strata.to_string(index=False))

print(
    f"\nCasos obrigatorios por Severe raro: "
    f"{int(sample['mandatory_rare_severe'].sum())}"
)

print(
    f"Estudos com algum rotulo ausente: "
    f"{int(sample[target_cols].isna().any(axis=1).sum())}"
)

# ---------------------------------------------------------------------
# 9. EXPORTAR
# ---------------------------------------------------------------------

print("\n===== 9. EXPORTANDO =====")

sample_ids = set(sample["study_id"])

sample_studies = sample[
    [
        "study_id",
        "study_stratum",
        "mandatory_rare_severe",
    ]
].copy()

sample_train = train[
    train["study_id"].isin(sample_ids)
].copy()

sample_series = series[
    series["study_id"].isin(sample_ids)
].copy()

sample_coords = coords[
    coords["study_id"].isin(sample_ids)
].copy()

sample_studies.to_csv(
    PROCESSED / "sample_studies.csv",
    index=False,
)

sample_train.to_csv(
    PROCESSED / "sample_train.csv",
    index=False,
)

sample_series.to_csv(
    PROCESSED / "sample_series_descriptions.csv",
    index=False,
)

sample_coords.to_csv(
    PROCESSED / "sample_label_coordinates.csv",
    index=False,
)

coverage_df.to_csv(
    TABLES / "sample_class_coverage.csv",
    index=False,
)

comparison.to_csv(
    TABLES / "sample_distribution_comparison.csv",
    index=False,
)

sample_strata.to_csv(
    TABLES / "sample_study_strata.csv",
    index=False,
)

metadata = {
    "seed": SEED,
    "sample_size": SAMPLE_SIZE,
    "total_studies": int(len(train)),
    "eligible_studies": int(len(eligible)),
    "mandatory_rare_severe_studies": int(
        sample["mandatory_rare_severe"].sum()
    ),
    "rare_severe_threshold": 20,
    "eligibility": [
        "study_id presente em train_label_coordinates.csv",
        "presenca de Sagittal T1",
        "presenca de Sagittal T2/STIR",
        "presenca de Axial T2",
    ],
}

with open(
    PROCESSED / "sample_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("Arquivos processados:")

for path in sorted(PROCESSED.glob("*")):
    print(f" - {path.name} | {path.stat().st_size} bytes")

print("\n===== MODULO 05 CONCLUIDO =====")
