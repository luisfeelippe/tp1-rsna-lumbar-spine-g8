"""
Motor de experimentos (Samuel) - usado pelos scripts 11 (grade) e 12 (ablacao).

Formulacao da tarefa
--------------------
Unidade de amostra: uma ROI = (study_id, condicao, nivel). Entrada: vetor de
descritores hand-crafted da ROI (+ one-hot do nivel). Saida: severidade em
{Normal/Mild, Moderate, Severe}. Treina-se UM modelo por condicao (5 modelos),
porque cada condicao e anotada em uma serie diferente (Sag T2/STIR, Sag T1, Ax T2)
e a distribuicao dos descritores muda entre elas.

Protocolo (sem vazamento)
-------------------------
* Folds externos: data/processed/folds.csv (Luis), congelados, por study_id.
* Ajuste de hiperparametros: GridSearchCV DENTRO do treino de cada fold externo,
  com StratifiedGroupKFold(3) agrupado por study_id (CV aninhada).
* Imputacao, padronizacao e PCA ficam dentro do Pipeline -> ajustados so no treino.
* Os ~23 rotulos sem coordenada (sem ROI) recebem a prevalencia do treino do fold
  (mesma regra do baseline), para que todos os 12.387 rotulos sejam avaliados.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from comum import (CONDITIONS, LABELS, LEVELS, N_SPLITS, PROBA_COLS, PROCESSED, SEED,
                   adicionar_target, carregar_rotulos_longos)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PREFIXOS = {"GLCM": "glcm_", "LBP": "lbp_", "HOG": "hog_", "INT": "int_"}
NIVEL_COLS = [f"nivel_{lv.replace('/', '_')}" for lv in LEVELS]
LIMIAR_PCA = 50  # conjuntos com mais colunas que isso passam por PCA (95% var.)


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
def carregar_tabela():
    """Junta rotulos (formato longo) + descritores disponiveis + one-hot de nivel."""
    feats = pd.read_csv(PROCESSED / "features_glcm_lbp.csv")
    hog_path = PROCESSED / "features_hog_intensidade.csv.gz"
    if hog_path.exists():
        hog = pd.read_csv(hog_path)
        feats = feats.merge(hog, on=["study_id", "series_id", "instance_number",
                                     "condition", "level"], how="inner",
                            validate="one_to_one")
    feats = adicionar_target(feats)
    for lv, col in zip(LEVELS, NIVEL_COLS):
        feats[col] = (feats["level"] == lv).astype(float)
    rot = carregar_rotulos_longos()
    tabela = rot.merge(feats.drop(columns=["series_id", "instance_number"]),
                       on=["study_id", "target"], how="left")
    # condicao/nivel para linhas sem ROI (derivados do nome do alvo)
    faltando = tabela["condition"].isna()
    if faltando.any():
        mapa = {f"{c.lower().replace(' ', '_')}_{l.lower().replace('/', '_')}": (c, l)
                for c in CONDITIONS for l in LEVELS}
        tabela.loc[faltando, "condition"] = tabela.loc[faltando, "target"].map(
            lambda t: mapa[t][0])
        tabela.loc[faltando, "level"] = tabela.loc[faltando, "target"].map(
            lambda t: mapa[t][1])
    tabela["tem_roi"] = ~faltando
    return tabela


def conjuntos_disponiveis(tabela):
    disp = {}
    for nome, pre in PREFIXOS.items():
        cols = [c for c in tabela.columns if c.startswith(pre)]
        if cols:
            disp[nome] = cols
    if len(disp) >= 2:
        disp["TODOS"] = [c for n in disp for c in disp[n]]
    return disp


# ----------------------------------------------------------------------------
# Modelos e grades
# ----------------------------------------------------------------------------
def _grade(modelo, rapido):
    if modelo == "LR":
        return {"clf__C": [0.1, 1.0] if rapido else [0.01, 0.1, 1.0, 10.0]}
    if modelo == "SVM":
        return ({"clf__C": [1.0], "clf__gamma": ["scale"]} if rapido else
                {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01]})
    if modelo == "RF":
        return ({"clf__min_samples_leaf": [5]} if rapido else
                {"clf__min_samples_leaf": [1, 5]})
    if modelo == "HGB":
        return ({"clf__learning_rate": [0.1]} if rapido else
                {"clf__max_depth": [3, 6]})
    if modelo == "KNN":
        return {"clf__n_neighbors": [15] if rapido else [5, 15, 31]}
    raise ValueError(modelo)


def _classificador(modelo, usar_peso, rapido):
    cw = "balanced" if usar_peso else None
    if modelo == "LR":
        return LogisticRegression(max_iter=3000, class_weight=cw, random_state=SEED)
    if modelo == "SVM":
        return SVC(kernel="rbf", class_weight=cw, random_state=SEED)
    if modelo == "RF":
        return RandomForestClassifier(n_estimators=150 if rapido else 200,
                                      class_weight="balanced_subsample" if usar_peso
                                      else None, random_state=SEED, n_jobs=1)
    if modelo == "HGB":
        return HistGradientBoostingClassifier(max_iter=150 if rapido else 200,
                                              learning_rate=0.1,
                                              class_weight=cw, early_stopping=False,
                                              random_state=SEED)
    if modelo == "KNN":  # kNN nao aceita pesos de classe; usado sem balanceamento
        return KNeighborsClassifier(weights="distance")
    raise ValueError(modelo)


def montar_pipeline(modelo, cols_img, usar_nivel=True, usar_peso=True,
                    escalonar=True, rapido=False, usar_pca=True):
    passos_img = [("imp", SimpleImputer(strategy="median"))]
    arvore = modelo in ("RF", "HGB")
    if escalonar and not arvore:
        passos_img.append(("esc", StandardScaler()))
    if usar_pca and not arvore and len(cols_img) > LIMIAR_PCA:
        passos_img.append(("pca", PCA(n_components=0.95, random_state=SEED)))
    transformadores = []
    if cols_img:
        transformadores.append(("img", Pipeline(passos_img), cols_img))
    if usar_nivel:
        transformadores.append(("nivel", "passthrough", NIVEL_COLS))
    pre = ColumnTransformer(transformadores, remainder="drop")
    return Pipeline([("pre", pre), ("clf", _classificador(modelo, usar_peso, rapido))])


# ----------------------------------------------------------------------------
# Execucao
# ----------------------------------------------------------------------------
def _prior(y):
    cont = pd.Series(y).value_counts().reindex(LABELS, fill_value=0).to_numpy(float)
    return cont / cont.sum()


def rodar(tabela, cols_img, modelo, usar_nivel=True, usar_peso=True, escalonar=True,
          rapido=False, n_jobs=-1, verbose=True, usar_pca=True):
    """Executa 5 condicoes x 5 folds. Retorna (oof DataFrame, hiperparametros)."""
    oof, params = [], []
    cols_x = list(cols_img) + (NIVEL_COLS if usar_nivel else [])
    for cond in CONDITIONS:
        dc = tabela[tabela["condition"] == cond]
        for fold in range(N_SPLITS):
            tr = dc[(dc["fold"] != fold) & dc["tem_roi"]]
            va = dc[dc["fold"] == fold]
            va_roi, va_sem = va[va["tem_roi"]], va[~va["tem_roi"]]

            pipe = montar_pipeline(modelo, list(cols_img), usar_nivel, usar_peso,
                                   escalonar, rapido, usar_pca)
            cv_int = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
            busca = GridSearchCV(pipe, _grade(modelo, rapido), scoring="f1_macro",
                                 cv=cv_int, n_jobs=n_jobs, refit=False,
                                 error_score="raise")
            busca.fit(tr[cols_x], tr["y_true"], groups=tr["study_id"])
            melhor = busca.best_params_
            final = clone(pipe).set_params(**melhor)
            if modelo == "SVM":
                final.set_params(clf__probability=True)
            final.fit(tr[cols_x], tr["y_true"])

            proba = final.predict_proba(va_roi[cols_x])
            ordem = [list(final.classes_).index(l) for l in LABELS]
            proba = proba[:, ordem]
            pred = final.predict(va_roi[cols_x])
            bloco = va_roi[["study_id", "fold", "target", "y_true"]].copy()
            bloco["y_pred"] = pred
            bloco[PROBA_COLS] = proba
            bloco["fonte"] = "modelo"
            oof.append(bloco)

            # rotulos sem ROI -> prevalencia do treino do alvo (regra do baseline)
            for alvo, g in va_sem.groupby("target"):
                ytr = tabela[(tabela["target"] == alvo) & (tabela["fold"] != fold)][
                    "y_true"]
                pr = _prior(ytr)
                b = g[["study_id", "fold", "target", "y_true"]].copy()
                b["y_pred"] = LABELS[int(np.argmax(pr))]
                b[PROBA_COLS] = np.tile(pr, (len(g), 1))
                b["fonte"] = "prior_sem_roi"
                oof.append(b)

            params.append({"condition": cond, "fold": fold,
                           "cv_interno_f1_macro": busca.best_score_,
                           **{k.replace("clf__", ""): v for k, v in melhor.items()}})
            if verbose:
                print(f"   {cond:<34} fold {fold} | f1_macro(CV interna)="
                      f"{busca.best_score_:.3f} | {melhor}")
    oof = pd.concat(oof, ignore_index=True)
    assert not oof.duplicated(["study_id", "target"]).any()
    return oof, pd.DataFrame(params)
