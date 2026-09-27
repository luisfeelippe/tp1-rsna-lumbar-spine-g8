from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = RAW / "train.csv"
SERIES_PATH = RAW / "train_series_descriptions.csv"
COORD_PATH = RAW / "train_label_coordinates.csv"

LABEL_ORDER = ["Normal/Mild", "Moderate", "Severe"]
SERIES_EXPECTED = ["Sagittal T1", "Sagittal T2/STIR", "Axial T2"]

print("=" * 72)
print("TP1 RSNA G8 - MODULO 04")
print("EDA - ANALISE EXPLORATORIA DOS ROTULOS")
print("=" * 72)

train = pd.read_csv(TRAIN_PATH)
series = pd.read_csv(SERIES_PATH)
coords = pd.read_csv(COORD_PATH)

target_cols = [c for c in train.columns if c != "study_id"]

# ---------------------------------------------------------------------
# 1. FORMATO LONGO DOS ROTULOS
# ---------------------------------------------------------------------

long = train.melt(
    id_vars="study_id",
    value_vars=target_cols,
    var_name="target",
    value_name="severity",
)

print("\n===== 1. DISTRIBUICAO GLOBAL =====")

valid = long.dropna(subset=["severity"]).copy()

global_dist = (
    valid["severity"]
    .value_counts()
    .reindex(LABEL_ORDER, fill_value=0)
    .rename_axis("severity")
    .reset_index(name="count")
)

global_dist["percentage_valid_labels"] = (
    global_dist["count"] / len(valid) * 100
)

print(global_dist.to_string(index=False))

global_dist.to_csv(
    TABLES / "eda_global_class_distribution.csv",
    index=False,
)

# ---------------------------------------------------------------------
# 2. DISTRIBUICAO POR ALVO
# ---------------------------------------------------------------------

print("\n===== 2. DISTRIBUICAO POR ALVO =====")

per_target = (
    valid.groupby(["target", "severity"], observed=True)
    .size()
    .unstack(fill_value=0)
    .reindex(columns=LABEL_ORDER, fill_value=0)
)

per_target_pct = (
    per_target.div(per_target.sum(axis=1), axis=0) * 100
)

per_target_out = per_target.copy()

for label in LABEL_ORDER:
    per_target_out[f"{label}_pct"] = per_target_pct[label]

per_target_out = per_target_out.reset_index()

per_target_out.to_csv(
    TABLES / "eda_distribution_by_target.csv",
    index=False,
)

severe_counts = per_target["Severe"].sort_values()

print("\nMenores contagens da classe Severe:")
print(severe_counts.head(10).to_string())

print("\nMaiores contagens da classe Severe:")
print(severe_counts.tail(10).sort_values(ascending=False).to_string())

# ---------------------------------------------------------------------
# 3. PERFIL DE SEVERIDADE POR ESTUDO
# ---------------------------------------------------------------------

print("\n===== 3. PERFIL POR ESTUDO =====")

profile = pd.DataFrame({"study_id": train["study_id"]})

for label in LABEL_ORDER:
    profile[label.lower().replace("/", "_").replace(" ", "_")] = (
        train[target_cols].eq(label).sum(axis=1)
    )

profile["missing"] = train[target_cols].isna().sum(axis=1)
profile["valid_labels"] = len(target_cols) - profile["missing"]

profile.to_csv(
    TABLES / "study_severity_profile.csv",
    index=False,
)

print(profile.describe().to_string())

# ---------------------------------------------------------------------
# 4. COBERTURA DAS SERIES
# ---------------------------------------------------------------------

print("\n===== 4. COBERTURA DAS SERIES =====")

coverage = (
    series.assign(present=1)
    .pivot_table(
        index="study_id",
        columns="series_description",
        values="present",
        aggfunc="max",
        fill_value=0,
    )
)

for desc in SERIES_EXPECTED:
    if desc not in coverage.columns:
        coverage[desc] = 0

coverage = coverage[SERIES_EXPECTED].copy()

coverage.columns = [
    "has_sagittal_t1",
    "has_sagittal_t2_stir",
    "has_axial_t2",
]

coverage = coverage.astype(int)
coverage["required_series_present"] = coverage.sum(axis=1)
coverage = coverage.reset_index()

series_counts = (
    series.groupby("study_id")
    .size()
    .rename("series_count")
    .reset_index()
)

coverage = coverage.merge(
    series_counts,
    on="study_id",
    how="left",
)

coverage.to_csv(
    TABLES / "study_series_coverage.csv",
    index=False,
)

coverage_summary = (
    coverage["required_series_present"]
    .value_counts()
    .sort_index()
    .rename_axis("required_series_present")
    .reset_index(name="studies")
)

print("\nQuantidade das 3 descricoes principais presente por estudo:")
print(coverage_summary.to_string(index=False))

complete_three = int((coverage["required_series_present"] == 3).sum())

print(
    f"\nEstudos contendo as tres descricoes principais: "
    f"{complete_three}/{len(coverage)} "
    f"({complete_three / len(coverage) * 100:.2f}%)"
)

# ---------------------------------------------------------------------
# 5. ESTUDOS SEM COORDENADAS
# ---------------------------------------------------------------------

print("\n===== 5. ESTUDOS SEM COORDENADAS =====")

train_ids = set(train["study_id"])
coord_ids = set(coords["study_id"])

without_coords = sorted(train_ids - coord_ids)

without_coords_df = train[
    train["study_id"].isin(without_coords)
].copy()

without_coords_df.to_csv(
    TABLES / "studies_without_coordinates.csv",
    index=False,
)

print(f"Quantidade: {len(without_coords)}")

if without_coords:
    print("study_id:")
    for study_id in without_coords:
        print(f" - {study_id}")

# ---------------------------------------------------------------------
# 6. AUSENCIA DE ROTULOS POR ESTUDO
# ---------------------------------------------------------------------

print("\n===== 6. AUSENCIA DE ROTULOS POR ESTUDO =====")

missing_studies = (
    profile[profile["missing"] > 0]
    .sort_values(["missing", "study_id"], ascending=[False, True])
)

missing_studies.to_csv(
    TABLES / "studies_with_missing_labels.csv",
    index=False,
)

print(f"Estudos com pelo menos um rotulo ausente: {len(missing_studies)}")
print(
    f"Maior quantidade de rotulos ausentes em um unico estudo: "
    f"{int(profile['missing'].max())}"
)

# ---------------------------------------------------------------------
# 7. FIGURA - DISTRIBUICAO GLOBAL
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    global_dist["severity"],
    global_dist["count"],
)

ax.set_title("Distribuição global das classes de severidade")
ax.set_xlabel("Classe")
ax.set_ylabel("Quantidade de rótulos válidos")

for i, row in global_dist.iterrows():
    ax.text(
        i,
        row["count"],
        f"{row['percentage_valid_labels']:.1f}%",
        ha="center",
        va="bottom",
    )

fig.tight_layout()

fig.savefig(
    FIGURES / "01_distribuicao_global_classes.png",
    dpi=180,
)

plt.close(fig)

# ---------------------------------------------------------------------
# 8. FIGURA - SERIES
# ---------------------------------------------------------------------

series_dist = (
    series["series_description"]
    .value_counts()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    series_dist.index,
    series_dist.values,
)

ax.set_title("Distribuição das séries de ressonância")
ax.set_xlabel("Descrição da série")
ax.set_ylabel("Quantidade de séries")

ax.tick_params(axis="x", rotation=15)

fig.tight_layout()

fig.savefig(
    FIGURES / "02_distribuicao_series.png",
    dpi=180,
)

plt.close(fig)

# ---------------------------------------------------------------------
# 9. FIGURA - MAPA DE PERCENTUAIS POR ALVO
# ---------------------------------------------------------------------

plot_pct = per_target_pct.copy()

pretty_names = [
    name.replace("_", " ")
    for name in plot_pct.index
]

fig, ax = plt.subplots(figsize=(9, 13))

im = ax.imshow(
    plot_pct[LABEL_ORDER].to_numpy(),
    aspect="auto",
)

ax.set_xticks(range(len(LABEL_ORDER)))
ax.set_xticklabels(LABEL_ORDER)

ax.set_yticks(range(len(pretty_names)))
ax.set_yticklabels(pretty_names, fontsize=7)

ax.set_title("Percentual das classes por alvo clínico")

for i in range(len(plot_pct)):
    for j in range(len(LABEL_ORDER)):
        value = plot_pct.iloc[i, j]
        ax.text(
            j,
            i,
            f"{value:.0f}%",
            ha="center",
            va="center",
            fontsize=6,
        )

fig.colorbar(im, ax=ax, label="Percentual (%)")

fig.tight_layout()

fig.savefig(
    FIGURES / "03_percentual_classes_por_alvo.png",
    dpi=180,
)

plt.close(fig)

# ---------------------------------------------------------------------
# 10. FIGURA - MISSING POR ALVO
# ---------------------------------------------------------------------

missing_target = (
    train[target_cols]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

missing_target = missing_target[missing_target > 0]

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    [
        x.replace("_", " ")
        for x in missing_target.index[::-1]
    ],
    missing_target.values[::-1],
)

ax.set_title("Rótulos ausentes por alvo clínico")
ax.set_xlabel("Quantidade de valores ausentes")

ax.tick_params(axis="y", labelsize=7)

fig.tight_layout()

fig.savefig(
    FIGURES / "04_rotulos_ausentes_por_alvo.png",
    dpi=180,
)

plt.close(fig)

# ---------------------------------------------------------------------
# 11. VERIFICACAO DOS ARTEFATOS
# ---------------------------------------------------------------------

print("\n===== 7. ARTEFATOS GERADOS =====")

for path in sorted(TABLES.glob("*.csv")):
    print(f"TABELA: {path.name} | {path.stat().st_size} bytes")

for path in sorted(FIGURES.glob("*.png")):
    print(f"FIGURA: {path.name} | {path.stat().st_size} bytes")

print("\n===== MODULO 04 CONCLUIDO =====")
