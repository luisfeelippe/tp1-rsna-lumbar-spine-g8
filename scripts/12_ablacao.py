"""
TP1 RSNA G8 - Script 12 (Samuel)
Estudo de ablacao: qual etapa do pipeline mais contribui?

Parte do melhor par (conjunto, modelo) da grade do script 11 (por F1 macro) e
remove UMA etapa por vez, mantendo todo o resto identico (mesmos folds, mesma
CV aninhada, mesma grade de hiperparametros):

  A  pipeline completo (referencia)
  B  sem pesos de classe (sem tratamento de desbalanceamento)
  C  sem one-hot do nivel vertebral (so descritores de imagem)
  D  so o nivel vertebral (nenhum descritor de imagem) -> quanto sinal vem da imagem?
  E  sem padronizacao z-score          (so se o modelo for LR/SVM)
  F  sem PCA                           (so se houver PCA no pipeline)

A comparacao entre familias isoladas e combinadas (GLCM vs LBP vs HOG vs INT vs
TODOS) ja esta na grade do script 11 e entra na discussao junto com esta tabela.

Uso:
    python scripts/12_ablacao.py                       # usa o melhor da grade
    python scripts/12_ablacao.py --conjunto TODOS --modelo HGB
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from comum import METRICAS, TABLES, avaliar_oof, fmt  # noqa: E402
from experimentos import (LIMIAR_PCA, carregar_tabela, conjuntos_disponiveis,  # noqa: E402
                          rodar)

ap = argparse.ArgumentParser()
ap.add_argument("--conjunto")
ap.add_argument("--modelo")
ap.add_argument("--rapido", action="store_true")
ap.add_argument("--n-jobs", type=int, default=-1)
args = ap.parse_args()

grade = pd.read_csv(TABLES / "grade_resultados.csv")
modelos_grade = grade[grade["conjunto"] != "—"]
if args.conjunto and args.modelo:
    conj, mod = args.conjunto, args.modelo
else:
    melhor = modelos_grade.sort_values("f1_macro_mean", ascending=False).iloc[0]
    conj, mod = melhor["conjunto"], melhor["modelo"]

tabela = carregar_tabela()
cols = conjuntos_disponiveis(tabela)[conj]
print("=" * 72)
print(f"TP1 RSNA G8 - SCRIPT 12 | ABLACAO sobre {conj} + {mod}")
print("=" * 72)

variantes = [
    ("A", "Pipeline completo", dict()),
    ("B", "Sem pesos de classe", dict(usar_peso=False)),
    ("C", "Sem nível vertebral (só imagem)", dict(usar_nivel=False)),
    ("D", "Só nível vertebral (sem imagem)", dict(cols_img=[])),
]
if mod in ("LR", "SVM", "KNN"):
    variantes.append(("E", "Sem padronização z-score", dict(escalonar=False)))
    if len(cols) > LIMIAR_PCA:
        variantes.append(("F", "Sem PCA", dict(usar_pca=False)))

linhas = []
for cod, nome, kw in variantes:
    cols_img = kw.pop("cols_img", cols)
    print(f"\n>>> [{cod}] {nome}")
    oof, _ = rodar(tabela, cols_img, mod, rapido=args.rapido, n_jobs=args.n_jobs,
                   verbose=False, **kw)
    _, _, res = avaliar_oof(oof)
    linha = {"variante": cod, "descricao": nome}
    for _, r in res.iterrows():
        linha[f"{r.metric}_mean"], linha[f"{r.metric}_std"] = r["mean"], r["std"]
    linhas.append(linha)
    print(f"    F1 macro = {fmt(linha['f1_macro_mean'], linha['f1_macro_std'])}")

abl = pd.DataFrame(linhas)
ref = abl.loc[abl.variante == "A", "f1_macro_mean"].iloc[0]
abl["delta_f1_macro_vs_A"] = abl["f1_macro_mean"] - ref
abl.insert(0, "base", f"{conj} + {mod}")
abl.to_csv(TABLES / "ablacao.csv", index=False)

f = abl[["variante", "descricao"]].copy()
for m in METRICAS:
    f[m] = [fmt(a, b) for a, b in zip(abl[f"{m}_mean"], abl[f"{m}_std"])]
f["delta_f1_macro_vs_A"] = abl["delta_f1_macro_vs_A"].map(lambda x: f"{x:+.4f}")
f.to_csv(TABLES / "ablacao_formatada.csv", index=False)
print("\n" + f[["variante", "descricao", "balanced_accuracy", "f1_macro",
                "delta_f1_macro_vs_A"]].to_string(index=False))
