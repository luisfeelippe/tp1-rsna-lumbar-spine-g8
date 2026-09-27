# TP1 — RSNA 2024 Lumbar Spine Degenerative Classification
## Contribuição técnica — Luis Felipe Xavier Falcão

### Escopo desta contribuição

Esta pasta contém a contribuição técnica de Luis Felipe Xavier Falcão para
o Trabalho Prático 1 da disciplina Tópicos Especiais em Sistemas de Informação.

Foram implementadas as seguintes atividades:

1. inspeção e validação dos dados tabulares;
2. análise exploratória dos rótulos;
3. caracterização do desbalanceamento;
4. definição de amostra reproduzível;
5. geração de folds de validação cruzada no nível de study_id;
6. validação de ausência de sobreposição entre folds;
7. baseline trivial de classe majoritária;
8. métricas e previsões out-of-fold.

## Dataset

Desafio:

RSNA 2024 Lumbar Spine Degenerative Classification

Arquivos tabulares utilizados:

- train.csv
- train_series_descriptions.csv
- train_label_coordinates.csv

Os dados brutos da competição não fazem parte deste pacote.

### Estrutura encontrada

- 1.975 estudos em train.csv;
- 25 alvos clínicos;
- 5 condições;
- 5 níveis lombares;
- 3 classes: Normal/Mild, Moderate e Severe;
- 6.294 séries;
- 48.692 anotações de coordenadas.

A distribuição global original dos rótulos válidos apresentou forte
desbalanceamento, com predominância de Normal/Mild.

## Amostra utilizada

Semente aleatória:

42

Tamanho:

500 estudos

Critérios de elegibilidade:

- study_id presente nas anotações de coordenadas;
- presença de Sagittal T1;
- presença de Sagittal T2/STIR;
- presença de Axial T2.

Foram encontrados 1.972 estudos elegíveis.

Para evitar o desaparecimento de classes extremamente raras, foram incluídos
obrigatoriamente todos os estudos contendo Severe em alvos com no máximo
20 ocorrências Severe entre os estudos elegíveis.

Isso resultou em 50 estudos obrigatórios.

As demais vagas foram preenchidas de forma reproduzível e estratificada
segundo o maior grau de severidade observado no estudo.

A amostra final apresentou:

- 500 estudos;
- 59 no estrato Normal/Mild;
- 174 no estrato Moderate;
- 267 no estrato Severe;
- 50 estudos obrigatórios por Severe raro.

A estratégia enriquece deliberadamente a proporção de casos Severe e,
portanto, os resultados devem ser interpretados como resultados sobre esta
amostra experimental, e não como uma estimativa direta da prevalência do
dataset completo.

## Folds

Foram definidos 5 folds fixos:

- fold 0: 100 estudos;
- fold 1: 100 estudos;
- fold 2: 100 estudos;
- fold 3: 100 estudos;
- fold 4: 100 estudos.

Unidade de particionamento:

study_id

Todas as imagens e séries pertencentes ao mesmo study_id DEVEM herdar o
mesmo fold.

Não gerar novos splits.

Utilizar obrigatoriamente:

data/processed/folds.csv

Nos experimentos seguintes:

- fold 0 como validação e 1-4 como treino;
- fold 1 como validação e 0,2,3,4 como treino;
- fold 2 como validação e 0,1,3,4 como treino;
- fold 3 como validação e 0,1,2,4 como treino;
- fold 4 como validação e 0-3 como treino.

Quatro combinações alvo x classe extremamente raras não aparecem em todos
os cinco folds. Isso foi registrado como limitação e não deve ser corrigido
por duplicação artificial de exames.

## Baseline trivial

Para cada um dos 25 alvos e para cada fold:

1. foram usados os quatro folds de treino;
2. valores ausentes do alvo foram ignorados;
3. a classe majoritária foi determinada apenas no treino;
4. todos os exemplos da validação receberam essa classe;
5. probabilidades triviais foram obtidas pelas prevalências das classes no
   treino da respectiva dobra.

Foram produzidas 12.387 previsões out-of-fold, uma para cada rótulo válido
da amostra.

### Resultado médio entre os 5 folds

- Accuracy: 0.7562 ± 0.0170
- Balanced Accuracy: 0.3333 ± 0.0000
- F1 Macro: 0.2838 ± 0.0036
- F1 Weighted: 0.6608 ± 0.0232
- Log-loss: 0.6377 ± 0.0369

Balanced Accuracy foi calculada como macro recall sobre as três classes
fixas: Normal/Mild, Moderate e Severe.

O baseline evidencia o efeito do desbalanceamento: apesar da accuracy
relativamente alta, o modelo majoritário não possui capacidade real de
discriminar adequadamente as classes minoritárias.

## Para os próximos integrantes

A partir desta contribuição, a equipe deve:

1. usar exatamente os study_id definidos em sample_studies.csv;
2. baixar/processar apenas as imagens necessárias à amostra;
3. associar cada imagem ao fold por study_id;
4. NÃO criar um novo train_test_split;
5. implementar o pré-processamento de imagem;
6. extrair no mínimo três famílias de características hand-crafted;
7. treinar os modelos clássicos usando os mesmos folds;
8. ajustar hiperparâmetros apenas com dados de treino;
9. comparar os modelos ao baseline trivial aqui produzido;
10. reportar média e desvio-padrão entre folds;
11. documentar a estratégia de amostragem e suas limitações.

## Execução

Criar ambiente Python 3.12 e instalar:

pip install -r requirements-luis.txt

Os CSVs oficiais do Kaggle devem estar em:

data/raw/

Executar:

python scripts/01_inspecionar_dados.py
python scripts/02_eda.py
python scripts/03_gerar_amostra.py
python scripts/04_gerar_folds.py
python scripts/05_baseline_trivial.py
python scripts/06_auditoria_final.py

A saída final da auditoria deve conter:

TODAS_AS_VALIDACOES_OK
