"""
TP1 RSNA G8 - Script 11 (Samuel)
Grade de experimentos DESCRITOR x MODELO com CV aninhada, sobre os folds congelados.

Uso:
    python scripts/11_treinar_modelos.py                  # grade completa
    python scripts/11_treinar_modelos.py --rapido         # grades reduzidas (teste)
    python scripts/11_treinar_modelos.py --conjuntos GLCM LBP --modelos LR RF
    python scripts/11_treinar_modelos.py --forcar         # refaz mesmo com cache

Saidas (outputs/tables):
    oof/oof_<CONJUNTO>_<MODELO>.csv     previsoes out-of-fold (cache)
    grade_resultados.csv                media e desvio de cada metrica (+ baseline)
    grade_resultados_formatada.csv      tabela "media ± desvio" pronta para o artigo
    melhores_hiperparametros.csv        hiperparametros escolhidos em cada fold
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from comum import METRICAS, TABLES, avaliar_oof, carregar_baseline_oof, fmt  # noqa: E402
from experimentos import carregar_tabela, conjuntos_disponiveis, rodar  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--conjuntos", nargs="*", default=None)
ap.add_argument("--modelos", nargs="*", default=["LR", "SVM", "RF", "HGB"])
ap.add_argument("--rapido", action="store_true")
ap.add_argument("--forcar", action="store_true")
ap.add_argument("--n-jobs", type=int, default=-1)
args = ap.parse_args()

OOF_DIR = TABLES / "oof"
OOF_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("TP1 RSNA G8 - SCRIPT 11 | GRADE DESCRITOR x MODELO")
print("=" * 72)
tabela = carregar_tabela()
disp = conjuntos_disponiveis(tabela)
conjuntos = args.conjuntos or list(disp)
faltam = [c for c in conjuntos if c not in disp]
if faltam:
    sys.exit(f"Conjuntos indisponiveis: {faltam}. Disponiveis: {list(disp)}. "
             "Para HOG/INT rode antes o script 10.")
print(f"Rotulos avaliados: {len(tabela)} | com ROI: {int(tabela.tem_roi.sum())}")
print("Conjuntos:", {k: len(v) for k, v in disp.items() if k in conjuntos})
print("Modelos:", args.modelos, "| modo rapido" if args.rapido else "")

linhas, params_all = [], []
for conj in conjuntos:
    for mod in args.modelos:
        arq = OOF_DIR / f"oof_{conj}_{mod}.csv"
        arq_p = OOF_DIR / f"params_{conj}_{mod}.csv"
        if arq.exists() and not args.forcar:
            print(f"\n[cache] {conj} + {mod}")
            oof = pd.read_csv(arq)
            params = pd.read_csv(arq_p) if arq_p.exists() else pd.DataFrame()
        else:
            print(f"\n>>> {conj} ({len(disp[conj])} cols) + {mod}")
            t0 = time.time()
            oof, params = rodar(tabela, disp[conj], mod, rapido=args.rapido,
                                n_jobs=args.n_jobs)
            oof.to_csv(arq, index=False)
            params.to_csv(arq_p, index=False)
            print(f"    tempo: {time.time() - t0:.0f}s")
        _, _, res = avaliar_oof(oof)
        linha = {"conjunto": conj, "modelo": mod}
        for _, r in res.iterrows():
            linha[f"{r.metric}_mean"] = r["mean"]
            linha[f"{r.metric}_std"] = r["std"]
        linhas.append(linha)
        if len(params):
            params_all.append(params.assign(conjunto=conj, modelo=mod))
        print(f"    F1 macro = {fmt(linha['f1_macro_mean'], linha['f1_macro_std'])} | "
              f"Bal. acc = {fmt(linha['balanced_accuracy_mean'], linha['balanced_accuracy_std'])}")

# baseline trivial (Luis) avaliado com o MESMO codigo de metricas
_, _, rb = avaliar_oof(carregar_baseline_oof())
base = {"conjunto": "—", "modelo": "Baseline trivial (classe majoritária)"}
for _, r in rb.iterrows():
    base[f"{r.metric}_mean"], base[f"{r.metric}_std"] = r["mean"], r["std"]

grade = pd.DataFrame([base] + linhas)
# junta com execucoes anteriores de outros conjuntos/modelos (se houver)
arq_grade = TABLES / "grade_resultados.csv"
if arq_grade.exists():
    antiga = pd.read_csv(arq_grade)
    chave = set(zip(grade.conjunto, grade.modelo))
    antiga = antiga[[(c, m) not in chave for c, m in zip(antiga.conjunto, antiga.modelo)]]
    grade = pd.concat([grade.iloc[:1], antiga[antiga.conjunto != "—"], grade.iloc[1:]],
                      ignore_index=True)
grade.to_csv(arq_grade, index=False)

fmt_tab = grade[["conjunto", "modelo"]].copy()
for m in METRICAS:
    fmt_tab[m] = [fmt(a, b) for a, b in zip(grade[f"{m}_mean"], grade[f"{m}_std"])]
fmt_tab.to_csv(TABLES / "grade_resultados_formatada.csv", index=False)
if params_all:
    pd.concat(params_all).to_csv(TABLES / "melhores_hiperparametros.csv", index=False)

print("\n" + "=" * 72)
print(fmt_tab[["conjunto", "modelo", "balanced_accuracy", "f1_macro",
               "log_loss_ponderada"]].to_string(index=False))
melhor = grade.iloc[1:].sort_values("f1_macro_mean", ascending=False).iloc[0]
print(f"\nMelhor por F1 macro: {melhor.conjunto} + {melhor.modelo} "
      f"({melhor.f1_macro_mean:.4f})")
print("=" * 72)
