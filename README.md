# TP1 — RSNA 2024 Lumbar Spine Degenerative Classification (Grupo G8)

Projeto desenvolvido para a disciplina de Tópicos Especiais em Sistemas de Informação. O objetivo é construir um *baseline* focado em visão computacional e aprendizado de máquina clássico para classificar condições degenerativas da coluna lombar, sem o uso de redes neurais profundas.

## Equipe (Grupo G8)
* **Luis Felipe Xavier Falcão:** Fundação de dados, amostragem, pipeline de validação e baseline trivial.
* **Gabriel Samilo Pinto de Oliveira:** Aquisição automatizada das imagens, pré-processamento DICOM, recorte de ROIs e engenharia de características (GLCM e LBP).
* **Samuel Brito da Silva:** Extração das famílias HOG e intensidade, grade descritor × modelo com CV aninhada, ajuste de hiperparâmetros, ablação, análise de erros e figuras.

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

**4. Reprodução da Fase 3 (3ª família, modelos, ablação e análise de erros):**

Requer a pasta `data/interim/rois/` (gerada pelo script 08).

```bash
# Famílias 3 e 4: HOG (gradiente) e intensidade de 1ª ordem
python scripts/10_extrair_hog_intensidade.py

# Grade descritor x modelo (LR, SVM, RF, HGB) com CV aninhada nos folds congelados
python scripts/11_treinar_modelos.py

# Estudo de ablação sobre a melhor configuração da grade
python scripts/12_ablacao.py

# Análise de erros + figuras 05-08 do artigo
python scripts/13_analise_erros_figuras.py
```

O script 11 guarda as previsões out-of-fold em `outputs/tables/oof/` e reaproveita esse cache
se for interrompido; use `--forcar` para refazer tudo e `--rapido` para um teste com grades reduzidas.

---

## Fase 3: Modelagem, Ablação e Análise de Erros

* **Descritores adicionais:** HOG (9 orientações, células 16×16, blocos 2×2, 324 valores) e intensidade de
  primeira ordem (média, desvio, assimetria, curtose, entropia, energia, percentis, IQR e contraste centro/borda,
  18 valores), extraídos das mesmas ROIs 64×64. Correção no script 09: histograma LBP com número fixo de bins (P+2).
* **Formulação:** unidade de amostra = ROI (study_id, condição, nível); saída = severidade em 3 classes.
  Um modelo por condição, com one-hot do nível vertebral como covariável.
* **Modelos:** Regressão Logística, SVM RBF, Random Forest e HistGradientBoosting, todos com pesos de classe
  balanceados. Imputação, padronização e PCA (95% da variância, conjuntos com mais de 50 colunas) ficam dentro do
  `Pipeline`, ajustados apenas no treino.
* **Protocolo:** folds externos de `folds.csv`; hiperparâmetros escolhidos por `GridSearchCV` (F1 macro) com
  `StratifiedGroupKFold(3)` agrupado por `study_id` dentro do treino de cada fold. Rótulos sem ROI recebem a
  prevalência do treino (mesma regra do baseline), para que os 12.387 rótulos sejam avaliados.
* **Métricas:** as mesmas do baseline, com a mesma agregação (fold × alvo → média nos 25 alvos → média ± desvio
  nos 5 folds), mais a log-loss ponderada da competição (pesos 1/2/4). Código comum em `scripts/comum.py`.

### Resultados da Fase 3 (média ± desvio nos 5 folds; reproduzidos com as versões do `requirements.txt`, Python 3.12 e 3.14)

F1 macro por conjunto de descritores e modelo (baseline trivial: 0,2838 ± 0,0036):

| Conjunto | LR | SVM | RF | HGB |
|---|---|---|---|---|
| GLCM (5) | 0,3730 ± 0,0181 | 0,3686 ± 0,0082 | 0,4070 ± 0,0071 | 0,3996 ± 0,0073 |
| LBP (10) | 0,3276 ± 0,0093 | 0,3358 ± 0,0205 | 0,3806 ± 0,0071 | 0,3788 ± 0,0112 |
| HOG (324) | 0,5068 ± 0,0125 | 0,5492 ± 0,0255 | 0,4659 ± 0,0120 | 0,5022 ± 0,0114 |
| INT (18) | 0,4843 ± 0,0187 | 0,4885 ± 0,0222 | 0,5005 ± 0,0141 | 0,4898 ± 0,0168 |
| TODOS (357) | 0,5418 ± 0,0224 | **0,5670 ± 0,0250** | 0,5004 ± 0,0098 | 0,5418 ± 0,0176 |

Tabela completa (todas as métricas, com desvios): `outputs/tables/grade_resultados_formatada.csv`.
Melhor configuração: **TODOS + SVM RBF** — F1 macro 0,5670, acurácia balanceada 0,5977 e log-loss
ponderada 0,6115 (baseline: 0,2838 / 0,3333 / 0,9720).

Ablação sobre TODOS + SVM (`outputs/tables/ablacao_formatada.csv`): sem padronização −0,217 de F1 macro;
sem pesos de classe −0,024 (acurácia balanceada −0,059); sem PCA −0,001; sem nível vertebral +0,002;
só nível vertebral, sem imagem: 0,209 (acaso).

Observação: Regressão Logística e SVM são determinísticos; RF e HGB podem variar na 4ª casa decimal
entre sistemas operacionais/versões de Python, por diferenças numéricas de baixo nível.

**Artefatos da Fase 3:** `outputs/tables/oof/` (previsões out-of-fold das 20 combinações),
`grade_resultados*.csv`, `melhores_hiperparametros.csv`, `ablacao*.csv`, `analise_*.csv`,
`casos_dificeis.csv` e as figuras `outputs/figures/05` a `08`.
