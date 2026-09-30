"""
TP1 RSNA G8 - Script 10 (Samuel)
Extracao das familias de descritores 3 e 4 sobre as MESMAS ROIs 64x64 do script 08:

  Familia 3 - Gradiente/bordas: HOG (Dalal & Triggs, 2005)
      9 orientacoes, celulas 16x16, blocos 2x2, normalizacao L2-Hys -> 324 valores.
      Motivacao: estenose e estreitamento alteram a GEOMETRIA local (contorno do canal,
      do forame e do disco); HOG codifica a distribuicao de orientacoes de borda, que
      as texturas GLCM/LBP nao descrevem explicitamente.

  Familia 4 - Intensidade de primeira ordem (histograma/momentos/percentis)
      Motivacao: a perda de sinal T2 do disco (desidratacao) e a reducao do liquor no
      canal alteram a distribuicao de intensidades da ROI (Michopoulou et al., 2011;
      van Griethuysen et al., 2017 - features first-order).

Saida: data/processed/features_hog_intensidade.csv.gz (uma linha por ROI, mesmas chaves
do features_glcm_lbp.csv).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from skimage import feature, img_as_ubyte, io

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ROI_DIR = ROOT / "data" / "interim" / "rois"
SAIDA = PROCESSED / "features_hog_intensidade.csv.gz"

HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                  block_norm="L2-Hys", feature_vector=True)
PERCENTIS = [5, 10, 25, 50, 75, 90, 95]
N_BINS_ENTROPIA = 32

print("=" * 72)
print("TP1 RSNA G8 - SCRIPT 10")
print("ENGENHARIA DE CARACTERISTICAS: HOG + INTENSIDADE (1a ORDEM)")
print("=" * 72)

meta_path = PROCESSED / "rois_metadata.csv"
if not meta_path.exists():
    sys.exit(f"ERRO: {meta_path} nao encontrado. Rode o script 08 antes.")
if not ROI_DIR.exists() or not any(ROI_DIR.iterdir()):
    sys.exit(f"ERRO: pasta de ROIs vazia/inexistente: {ROI_DIR.relative_to(ROOT)}.\n"
             "Rode o script 08 (precisa dos DICOMs) ou copie a pasta data/interim/rois.")

meta = pd.read_csv(meta_path)
print(f"ROIs listadas: {len(meta)}")


def intensidade(img):
    v = img.astype(np.float64).ravel() / 255.0
    hist, _ = np.histogram(v, bins=N_BINS_ENTROPIA, range=(0, 1))
    p = hist / hist.sum()
    p_pos = p[p > 0]
    feats = {
        "int_mean": v.mean(),
        "int_std": v.std(),
        "int_skew": stats.skew(v) if v.std() > 0 else 0.0,
        "int_kurtosis": stats.kurtosis(v) if v.std() > 0 else 0.0,
        "int_entropy": float(-(p_pos * np.log2(p_pos)).sum()),
        "int_energy": float((p ** 2).sum()),
        "int_min": v.min(),
        "int_max": v.max(),
    }
    pct = np.percentile(v, PERCENTIS)
    for q, val in zip(PERCENTIS, pct):
        feats[f"int_p{q}"] = val
    feats["int_iqr"] = pct[PERCENTIS.index(75)] - pct[PERCENTIS.index(25)]
    # razao centro/periferia: a regiao central 32x32 contem o alvo anotado
    c = img[16:48, 16:48].astype(np.float64) / 255.0
    feats["int_centro_mean"] = c.mean()
    feats["int_centro_menos_borda"] = c.mean() - v.mean()
    return feats


linhas = []
falhas = 0
for i, row in enumerate(meta.itertuples(index=False), start=1):
    caminho = ROI_DIR / row.roi_file
    try:
        img = img_as_ubyte(io.imread(caminho, as_gray=True))
        if img.shape != (64, 64):
            raise ValueError(f"shape inesperado {img.shape}")
        d = {"study_id": row.study_id, "series_id": row.series_id,
             "instance_number": row.instance_number,
             "condition": row.condition, "level": row.level}
        hog = feature.hog(img, **HOG_PARAMS)
        d.update({f"hog_{k}": v for k, v in enumerate(hog)})
        d.update(intensidade(img))
        linhas.append(d)
    except Exception as e:  # registra e segue
        falhas += 1
        print(f"Falha em {row.roi_file}: {e}")
    if i % 2000 == 0:
        print(f"[{i}/{len(meta)}] ROIs processadas...")

df = pd.DataFrame(linhas)
df.to_csv(SAIDA, index=False, compression="gzip")  # ~33 MB (limite do GitHub: 100 MB)
n_hog = sum(c.startswith("hog_") for c in df.columns)
n_int = sum(c.startswith("int_") for c in df.columns)
print("=" * 72)
print(f"CONCLUIDO: {len(df)} ROIs | {falhas} falhas")
print(f"HOG: {n_hog} valores | Intensidade: {n_int} valores")
print(f"Arquivo: {SAIDA.relative_to(ROOT)}")
print("=" * 72)
