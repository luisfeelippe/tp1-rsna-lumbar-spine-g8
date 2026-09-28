import subprocess
import sys
import zipfile
from pathlib import Path

# Configuração de caminhos baseada na estrutura do grupo
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMAGES_DIR = RAW / "train_images"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("TP1 RSNA G8 - MODULO 09")
print("VERIFICAÇÃO E EXTRAÇÃO DOS DICOMS DA AMOSTRA")
print("=" * 72)

KERNEL_SLUG = "gsamilo/notebookaf10d8d179"
ZIP_PATH = IMAGES_DIR / "amostra_g8_dicoms.zip"

# 1. Verificar se os DICOMs já estão extraídos
estudos_existentes = [p for p in IMAGES_DIR.iterdir() if p.is_dir()]
if len(estudos_existentes) >= 500:
    print(f"\nOK: Encontrados {len(estudos_existentes)} diretórios de exames em {IMAGES_DIR.relative_to(ROOT)}.")
    print("Os dados já estão prontos para o pré-processamento. Etapa concluída!")
    sys.exit(0)

# 2. Se o ZIP manual já estiver na pasta, apenas descompactar
if ZIP_PATH.exists():
    print(f"\nLocalizado arquivo {ZIP_PATH.name} ({ZIP_PATH.stat().st_size / (1024**3):.2f} GB).")
    print("Iniciando extração do arquivo compactado...")
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(IMAGES_DIR)
        
    print("Extração concluída com sucesso!")
    print(f"Diretórios extraídos em: {IMAGES_DIR.relative_to(ROOT)}")
    sys.exit(0)

# 3. Se não houver dados locais, tenta o download via API
print(f"\nBaixando o artefato do notebook {KERNEL_SLUG}...")
cmd_download = [
    "kaggle", "kernels", "output",
    KERNEL_SLUG,
    "-p", str(IMAGES_DIR)
]

result = subprocess.run(cmd_download, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

if result.returncode != 0:
    print("\n[ATENÇÃO] O download automático via API falhou devido ao tamanho do arquivo (~7.5GB).")
    print(f"Detalhe técnico: {result.stderr.strip()}")
    print("\nINSTRUÇÃO MANUAL DE REPRODUTIBILIDADE:")
    print(f"1. Acesse: https://www.kaggle.com/code/{KERNEL_SLUG}")
    print("2. Faça o download do arquivo 'amostra_g8_dicoms.zip' no painel Output.")
    print(f"3. Coloque o arquivo zip na pasta: {IMAGES_DIR.relative_to(ROOT)}")
    print("4. Execute este script novamente para descompactar automaticamente.")
    sys.exit(1)

# Extrair caso o download via CLI tenha sido bem-sucedido
if ZIP_PATH.exists():
    print("Download concluído. Extraindo arquivos...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(IMAGES_DIR)
    print("Processo finalizado com sucesso!")