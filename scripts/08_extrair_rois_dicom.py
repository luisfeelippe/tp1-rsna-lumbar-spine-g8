import pandas as pd
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 1. Configuração de Diretórios
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
IMAGES_DIR = RAW / "train_images"

# Criamos uma pasta "interim" (intermediária) para guardar as imagens recortadas (ROIs)
ROI_DIR = ROOT / "data" / "interim" / "rois"
ROI_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("TP1 RSNA G8 - MODULO 10")
print("PRÉ-PROCESSAMENTO: EXTRAÇÃO DE ROIs (REGIONS OF INTEREST)")
print("=" * 72)

# Tamanho do recorte ao redor da coordenada (64x64 pixels)
ROI_SIZE = 64
HALF_ROI = ROI_SIZE // 2

# 2. Carregar o mapeamento de coordenadas da amostra
coords_path = PROCESSED / "sample_label_coordinates.csv"
if not coords_path.exists():
    print(f"ERRO: Arquivo {coords_path.name} não encontrado.")
    sys.exit(1)

coords = pd.read_csv(coords_path)
print(f"Total de coordenadas a processar: {len(coords)}")

# 3. Função de Pré-processamento DICOM
def processar_dicom(dicom_path):
    #Lê o DICOM, aplica rescale slope/intercept e normaliza o contraste.
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(np.float32)
    
    # Aplica Slope e Intercept se existirem no cabeçalho DICOM
    slope = getattr(dcm, 'RescaleSlope', 1.0)
    intercept = getattr(dcm, 'RescaleIntercept', 0.0)
    img = img * slope + intercept
    
    # Normalização robusta baseada em percentis para remover artefatos de brilho extremo
    p_min, p_max = np.percentile(img, 1), np.percentile(img, 99)
    if p_max > p_min:
        img = np.clip((img - p_min) / (p_max - p_min), 0, 1)
    else:
        img = np.zeros_like(img)
        
    # Converte para escala de cinza 8-bits (0 a 255)
    return (img * 255).astype(np.uint8)

# 4. Processamento Iterativo
roi_metadata = []
sucessos = 0
erros = 0

print("Iniciando extração das ROIs... Isso pode levar alguns minutos.")

for idx, row in coords.iterrows():
    study = str(row['study_id'])
    series = str(row['series_id'])
    instance = str(row['instance_number'])
    
    # O caminho do arquivo DICOM exato
    dicom_path = IMAGES_DIR / study / series / f"{instance}.dcm"
    
    if not dicom_path.exists():
        erros += 1
        continue
        
    try:
        # Lê e normaliza a imagem
        img = processar_dicom(dicom_path)
        h, w = img.shape
        
        # Coordenadas centrais da anotação
        cx, cy = int(row['x']), int(row['y'])
        
        # Calcula os limites do Bounding Box garantindo que não ultrapassem as bordas da imagem
        y_min = max(0, cy - HALF_ROI)
        y_max = min(h, cy + HALF_ROI)
        x_min = max(0, cx - HALF_ROI)
        x_max = min(w, cx + HALF_ROI)
        
        roi = img[y_min:y_max, x_min:x_max]
        
        # Padronização de tamanho: preenche com preto (padding) caso o recorte fique na borda e seja menor que 64x64
        if roi.shape != (ROI_SIZE, ROI_SIZE):
            padded_roi = np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.uint8)
            padded_roi[0:roi.shape[0], 0:roi.shape[1]] = roi
            roi = padded_roi
            
        # Gera um nome único para o arquivo PNG
        nome_arquivo = f"{study}_{series}_{instance}_{row['condition']}_{row['level'].replace('/', '_')}.png"
        nome_arquivo = nome_arquivo.replace(" ", "_")
        caminho_salvar = ROI_DIR / nome_arquivo
        
        # Salva a ROI
        plt.imsave(caminho_salvar, roi, cmap='gray')
        
        # Registra no metadado
        roi_metadata.append({
            'study_id': study,
            'series_id': series,
            'instance_number': instance,
            'condition': row['condition'],
            'level': row['level'],
            'roi_file': nome_arquivo
        })
        
        sucessos += 1
        
        # Log de progresso a cada 1000 imagens
        if sucessos % 1000 == 0:
            print(f"[{sucessos}/{len(coords)}] ROIs processadas...")
            
    except Exception as e:
        erros += 1

# 5. Salvar tabela de metadados das ROIs geradas
df_roi = pd.DataFrame(roi_metadata)
df_roi.to_csv(PROCESSED / "rois_metadata.csv", index=False)

print("\n" + "=" * 72)
print(f"EXTRAÇÃO CONCLUÍDA: {sucessos} ROIs salvas | {erros} falhas.")
print(f"As imagens recortadas estão disponíveis em: data/interim/rois/")
print(f"Arquivo de mapeamento salvo em: data/processed/rois_metadata.csv")
print("=" * 72)