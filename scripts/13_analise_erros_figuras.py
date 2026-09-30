"""
TP1 RSNA G8 - Script 13 (Samuel)
Analise de erros + figuras finais do artigo.

Usa o melhor par (conjunto, modelo) por F1 macro da grade (script 11), ou o
informado por argumento, e compara com o baseline trivial.

Figuras (outputs/figures):
  05_grade_descritor_modelo.png   F1 macro e acuracia balanceada (media ± desvio) x baseline
  06_matriz_confusao.png          matriz de confusao do melhor modelo (contagem e % por linha)
  07_f1_por_condicao_nivel.png    F1 macro por condicao x nivel (melhor modelo)
  08_exemplos_erros.png           ROIs de acertos e falhas (so se data/interim/rois existir)
Tabelas (outputs/tables):
  analise_por_alvo.csv, analise_por_classe.csv, analise_por_condicao.csv,
  analise_por_nivel.csv, casos_dificeis.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from comum import (CONDITIONS, FIGURES, LABELS, LEVELS, PROCESSED, ROI_DIR,  # noqa: E402
                   TABLES, avaliar_oof, carregar_baseline_oof, target_name)

ap = argparse.ArgumentParser()
ap.add_argument("--conjunto")
ap.add_argument("--modelo")
args = ap.parse_args()
FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

grade = pd.read_csv(TABLES / "grade_resultados.csv")
mods = grade[grade.conjunto != "—"].copy()
if args.conjunto and args.modelo:
    conj, mod = args.conjunto, args.modelo
else:
    b = mods.sort_values("f1_macro_mean", ascending=False).iloc[0]
    conj, mod = b.conjunto, b.modelo
print(f"Melhor configuracao analisada: {conj} + {mod}")
oof = pd.read_csv(TABLES / "oof" / f"oof_{conj}_{mod}.csv")
base = carregar_baseline_oof()
base_row = grade[grade.conjunto == "—"].iloc[0]

# ---------------------------------------------------------------- Figura 05
ordem_conj = [c for c in ["GLCM", "LBP", "HOG", "INT", "TODOS"] if c in set(mods.conjunto)]
ordem_mod = [m for m in ["LR", "SVM", "RF", "HGB", "KNN"] if m in set(mods.modelo)]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
larg = 0.8 / len(ordem_mod)
for ax, met, titulo in [(axes[0], "f1_macro", "F1 macro"),
                        (axes[1], "balanced_accuracy", "Acurácia balanceada")]:
    x = np.arange(len(ordem_conj))
    for i, m in enumerate(ordem_mod):
        sub = mods[mods.modelo == m].set_index("conjunto").reindex(ordem_conj)
        ax.bar(x + i * larg - 0.4 + larg / 2, sub[f"{met}_mean"], larg,
               yerr=sub[f"{met}_std"], capsize=2, label=m)
    ax.axhline(base_row[f"{met}_mean"], color="k", ls="--", lw=1,
               label="Baseline trivial")
    ax.set_xticks(x, ordem_conj)
    ax.set_title(titulo)
    ax.set_ylim(0, max(0.6, mods[f"{met}_mean"].max() + 0.08))
    ax.grid(axis="y", alpha=0.3)
axes[0].legend(ncol=3, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES / "05_grade_descritor_modelo.png")
plt.close(fig)

# ---------------------------------------------------------------- Figura 06
cm = confusion_matrix(oof.y_true, oof.y_pred, labels=LABELS)
cm_pct = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.0%})", ha="center", va="center",
                color="white" if cm_pct[i, j] > 0.5 else "black", fontsize=8)
curto = ["Normal/Leve", "Moderado", "Grave"]
ax.set_xticks(range(3), curto)
ax.set_yticks(range(3), curto)
ax.set_xlabel("Predito")
ax.set_ylabel("Verdadeiro")
ax.set_title(f"{conj} + {mod} (OOF, 5 folds)")
fig.tight_layout()
fig.savefig(FIGURES / "06_matriz_confusao.png")
plt.close(fig)

# ---------------------------------------------------------------- Por alvo
pfa_m, _, _ = avaliar_oof(oof)
pfa_b, _, _ = avaliar_oof(base)
por_alvo = (pfa_m.groupby("target")[["f1_macro", "balanced_accuracy"]].mean()
            .join(pfa_b.groupby("target")[["f1_macro", "balanced_accuracy"]].mean(),
                  rsuffix="_baseline"))
cont = oof.groupby("target").y_true.value_counts().unstack(fill_value=0)
por_alvo = por_alvo.join(cont.reindex(columns=LABELS, fill_value=0))
por_alvo["ganho_f1_macro"] = por_alvo.f1_macro - por_alvo.f1_macro_baseline
por_alvo.to_csv(TABLES / "analise_por_alvo.csv")

mat = np.array([[por_alvo.loc[target_name(c, l), "f1_macro"] for l in LEVELS]
                for c in CONDITIONS])
fig, ax = plt.subplots(figsize=(5.2, 3.2))
im = ax.imshow(mat, cmap="viridis", vmin=0.25, vmax=max(0.6, mat.max()))
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                color="white" if mat[i, j] < 0.45 else "black", fontsize=8)
nomes = ["Canal central", "Forame esq.", "Forame dir.", "Subart. esq.", "Subart. dir."]
ax.set_xticks(range(5), LEVELS)
ax.set_yticks(range(5), nomes)
ax.set_title(f"F1 macro por condição × nível ({conj} + {mod})")
fig.colorbar(im, ax=ax, fraction=0.04)
fig.tight_layout()
fig.savefig(FIGURES / "07_f1_por_condicao_nivel.png")
plt.close(fig)

# ---------------------------------------------------------------- Tabelas extra
oof["condition"] = oof.target.map({target_name(c, l): c for c in CONDITIONS for l in LEVELS})
oof["level"] = oof.target.map({target_name(c, l): l for c in CONDITIONS for l in LEVELS})
por_classe = []
for nome, df in [("modelo", oof), ("baseline", base)]:
    cmx = confusion_matrix(df.y_true, df.y_pred, labels=LABELS)
    tot = cmx.sum()
    for i, lab in enumerate(LABELS):
        tp = cmx[i, i]
        fn = cmx[i].sum() - tp
        fp = cmx[:, i].sum() - tp
        tn = tot - tp - fn - fp
        por_classe.append({"fonte": nome, "classe": lab, "suporte": int(cmx[i].sum()),
                           "sensibilidade": tp / (tp + fn) if tp + fn else np.nan,
                           "especificidade": tn / (tn + fp) if tn + fp else np.nan,
                           "precisao": tp / (tp + fp) if tp + fp else np.nan})
pd.DataFrame(por_classe).to_csv(TABLES / "analise_por_classe.csv", index=False)
por_alvo2 = por_alvo.reset_index()
por_alvo2["condition"] = por_alvo2.target.map(oof.drop_duplicates("target")
                                              .set_index("target").condition)
por_alvo2["level"] = por_alvo2.target.map(oof.drop_duplicates("target")
                                          .set_index("target").level)
por_alvo2.groupby("condition")[["f1_macro", "f1_macro_baseline", "ganho_f1_macro"]] \
    .mean().to_csv(TABLES / "analise_por_condicao.csv")
por_alvo2.groupby("level")[["f1_macro", "f1_macro_baseline", "ganho_f1_macro"]] \
    .mean().to_csv(TABLES / "analise_por_nivel.csv")

# casos dificeis: Graves preditos como Normal/Leve com maior confianca
dif = oof[(oof.y_true == "Severe") & (oof.y_pred == "Normal/Mild")] \
    .sort_values("prob_normal_mild", ascending=False)
dif.to_csv(TABLES / "casos_dificeis.csv", index=False)

# ---------------------------------------------------------------- Figura 08
roi_meta_path = PROCESSED / "rois_metadata.csv"
if ROI_DIR.exists() and any(ROI_DIR.iterdir()) and roi_meta_path.exists():
    rm = pd.read_csv(roi_meta_path)
    rm["target"] = [target_name(c, l) for c, l in zip(rm.condition, rm.level)]
    o = oof[oof.get("fonte", "modelo") == "modelo"].merge(
        rm[["study_id", "target", "roi_file"]], on=["study_id", "target"])
    grupos = [
        ("Grave acertado", o[(o.y_true == "Severe") & (o.y_pred == "Severe")]
         .sort_values("prob_severe", ascending=False)),
        ("Grave → Normal/Leve", o[(o.y_true == "Severe") & (o.y_pred == "Normal/Mild")]
         .sort_values("prob_normal_mild", ascending=False)),
        ("Normal/Leve → Grave", o[(o.y_true == "Normal/Mild") & (o.y_pred == "Severe")]
         .sort_values("prob_severe", ascending=False)),
    ]
    n = 5
    curtos = {"Spinal Canal Stenosis": "Canal", "Left Neural Foraminal Narrowing": "Forame E",
              "Right Neural Foraminal Narrowing": "Forame D",
              "Left Subarticular Stenosis": "Subart. E",
              "Right Subarticular Stenosis": "Subart. D"}
    fig, axes = plt.subplots(3, n, figsize=(n * 1.3, 3 * 1.5))
    for r, (tit, g) in enumerate(grupos):
        g = g.drop_duplicates("condition").head(n) if len(g) >= n else g.head(n)
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(g):
                row = g.iloc[c]
                ax.imshow(plt.imread(ROI_DIR / row.roi_file), cmap="gray")
                ax.set_title(f"{curtos[row.condition]} {row.level}\n"
                             f"p(pred)={row[['prob_normal_mild', 'prob_moderate', 'prob_severe']].max():.2f}",
                             fontsize=6)
        axes[r, 0].text(-0.15, 0.5, tit, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "08_exemplos_erros.png")
    plt.close(fig)
    print("Figura 08 (exemplos de erros) gerada.")
else:
    print("AVISO: data/interim/rois nao encontrado -> figura 08 nao gerada.")

# ---------------------------------------------------------------- Resumo texto
pc = pd.DataFrame(por_classe)
print("\n===== RESUMO PARA O ARTIGO =====")
print(pc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nF1 macro medio por condicao:")
print(por_alvo2.groupby("condition")[["f1_macro", "f1_macro_baseline"]].mean().round(3))
print("\nF1 macro medio por nivel:")
print(por_alvo2.groupby("level")[["f1_macro", "f1_macro_baseline"]].mean().round(3))
print(f"\nGraves preditos como Normal/Leve: {len(dif)} de "
      f"{(oof.y_true == 'Severe').sum()} graves")
mod_err = oof[oof.y_true == "Moderate"].y_pred.value_counts(normalize=True)
print("Destino das previsoes para verdadeiros Moderados:\n", mod_err.round(3))
