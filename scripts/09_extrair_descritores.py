import pandas as pd
import numpy as np
import sys
from pathlib import Path
from skimage import io, feature, img_as_ubyte

# 1. Configuração de Diretórios
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ROI_DIR = ROOT / "data" / "interim" / "rois"

print("=" * 72)
print("TP1 RSNA G8 - MODULO 11")
print("ENGENHARIA DE CARACTERÍSTICAS: GLCM E LBP")
print("=" * 72)

# 2. Carregar o metadado das ROIs geradas no Script 08
roi_metadata_path = PROCESSED / "rois_metadata.csv"
if not roi_metadata_path.exists():
    print(f"ERRO: Arquivo {roi_metadata_path.name} não encontrado.")
    sys.exit(1)

metadata = pd.read_csv(roi_metadata_path)
print(f"Carregadas {len(metadata)} imagens para extração de características.")

# Parâmetros GLCM (distância de 1 e 3 pixels, ângulos de 0, 45, 90 e 135 graus)
DISTANCES = [1, 3]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

# Parâmetros LBP (raio 1, 8 pontos vizinhos, método uniforme)
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

dados_caracteristicas = []
sucessos = 0

print("Extraindo descritores manuais (Hand-crafted features)...")

for idx, row in metadata.iterrows():
    caminho_imagem = ROI_DIR / row['roi_file']
    
    if not caminho_imagem.exists():
        continue
        
    try:
        # Lê a imagem em escala de cinza
        img = io.imread(caminho_imagem, as_gray=True)
        img = img_as_ubyte(img) # Garante formato 8-bits
        
        features_linha = {
            'study_id': row['study_id'],
            'series_id': row['series_id'],
            'instance_number': row['instance_number'],
            'condition': row['condition'],
            'level': row['level']
        }
        
        # --- FAMÍLIA 1: TEXTURA CLÁSSICA (GLCM / HARALICK) ---
        glcm = feature.graycomatrix(img, distances=DISTANCES, angles=ANGLES, levels=256, symmetric=True, normed=True)
        for prop in GLCM_PROPS:
            # Calcula a propriedade e tira a média entre todas as distâncias e ângulos
            valor_prop = feature.graycoprops(glcm, prop).mean()
            features_linha[f'glcm_{prop}'] = valor_prop

        # --- FAMÍLIA 2: PADRÕES BINÁRIOS LOCAIS (LBP) ---
        lbp = feature.local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, method='uniform')
        # Calcula o histograma do LBP para virar um vetor fixo de features numéricas
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        for i, h_val in enumerate(hist):
            features_linha[f'lbp_bin_{i}'] = h_val
            
        dados_caracteristicas.append(features_linha)
        sucessos += 1
        
        if sucessos % 1000 == 0:
            print(f"[{sucessos}/{len(metadata)}] imagens processadas...")
            
    except Exception as e:
        print(f"Erro ao processar {row['roi_file']}: {e}")

# 3. Consolidar e Salvar
df_features = pd.DataFrame(dados_caracteristicas)
caminho_saida = PROCESSED / "features_glcm_lbp.csv"
df_features.to_csv(caminho_saida, index=False)

print("\n" + "=" * 72)
print(f"EXTRAÇÃO CONCLUÍDA: Descritores extraídos de {sucessos} imagens.")
print(f"Arquivo tabular gerado: {caminho_saida.relative_to(ROOT)}")
print(f"Total de características extraídas por imagem: {df_features.shape[1] - 5}") # -5 pelas colunas de ID
print("O trabalho está pronto para treinar os modelos!")
print("=" * 72)