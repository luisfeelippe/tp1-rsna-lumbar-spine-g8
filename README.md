# TP1 — RSNA 2024 Lumbar Spine Degenerative Classification (Grupo G8)

Projeto desenvolvido para a disciplina de Tópicos Especiais em Sistemas de Informação. O objetivo é construir um *baseline* focado em visão computacional e aprendizado de máquina clássico para classificar condições degenerativas da coluna lombar, sem o uso de redes neurais profundas.

## Equipe (Grupo G8)
* **Luis Felipe Xavier Falcão:** Fundação de dados, amostragem, pipeline de validação e baseline trivial.
* **Gabriel Samilo Pinto de Oliveira:** Aquisição automatizada das imagens, pré-processamento DICOM, recorte de ROIs e engenharia de características (GLCM e LBP).
* **Samuel Brito da Silva:** Treinamento de modelos clássicos, hiperparâmetros, extração da 3ª família de features e análise de erros.

---

## Fase 1: Fundação e Protocolo de Validação

A primeira etapa garantiu a criação de um ambiente seguro e livre de vazamento de dados (*data leakage*).

* **Amostragem:** Como o dataset original excede os limites de hardware, foi definida uma amostra reproduzível de 500 estudos (semente 42), preservando obrigatoriamente casos extremamente raros da classe *Severe*.


* **Validação Cruzada (Folds):** Foram criados 5 folds estratificados no nível de `study_id`. Todas as imagens de um mesmo paciente pertencem obrigatoriamente à mesma dobra de validação.


* **Baseline Trivial:** Implementou-se um classificador de classe majoritária calculado apenas nos dados de treino.
* *Métricas de Referência:* Accuracy: 0.7562 | Balanced Accuracy: 0.3333 | F1 Macro: 0.2838 | F1 Weighted: 0.6608 | Log-loss: 0.6377.



---

## Fase 2: Imagens e Engenharia de Características

A segunda etapa estabeleceu a ponte entre os metadados e os modelos através da visão computacional clássica.

* **Aquisição Reprodutível:** Para evitar o download massivo da competição, o dataset foi filtrado em nuvem via *Kaggle Kernel*, gerando um artefato consolidado de ~7.5 GB, baixado e extraído automaticamente via script.
* **Pré-processamento DICOM:** Leitura das imagens nativas com a biblioteca `pydicom`, aplicando correções de `RescaleSlope` e `RescaleIntercept`. O contraste foi normalizado de forma robusta cortando os percentis 1% e 99% para remoção de *outliers* e reflexos.
* **Extração de ROIs (Regions of Interest):** Com base nas coordenadas médicas fornecidas no dataset, foram isolados recortes exatos de 64x64 pixels focados diretamente na lesão (disco intervertebral ou forame neural), totalizando 12.369 ROIs purificadas.
* **Extração de Descritores Manuais (Hand-crafted Features):**
* **GLCM (Gray-Level Co-occurrence Matrix):** Extração de texturas clássicas (Contraste, Dissimilaridade, Homogeneidade, Energia e Correlação) para capturar o ressecamento do disco.
* **LBP (Local Binary Patterns):** Implementação do LBP uniforme invariante à rotação para mapear micro-rugosidades e calcificações ósseas.

Os resultados formaram um dataset tabular de 15 colunas numéricas de características salvo em `data/processed/features_glcm_lbp.csv`.

---

## Como Executar o Projeto

**1. Preparação do Ambiente:**

```bash
pip install -r requirements.txt

```

**2. Reprodução da Fase 1 (Tabelas e Folds):**

```bash
python scripts/01_inspecionar_dados.py
python scripts/02_eda.py
python scripts/03_gerar_amostra.py
python scripts/04_gerar_folds.py
python scripts/05_baseline_trivial.py
python scripts/06_auditoria_final.py

```

**3. Reprodução da Fase 2 (Imagens e Features):**
*Requisito: É necessário possuir as credenciais da API do Kaggle configuradas localmente (`kaggle.json`) e ter aceitado as regras da competição.*

```bash
# Baixa as imagens filtradas do Kaggle
python scripts/07_baixar_dicoms_amostra.py

# Recorta as áreas lesionadas das imagens DICOM
python scripts/08_extrair_rois_dicom.py

# Extrai as matrizes GLCM e LBP
python scripts/09_extrair_descritores.py

```

```



```