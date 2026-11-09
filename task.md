# Tarefas — Trabalho Prático de Mineração de Dados (Grupo 7)

Base: **Predict Students' Dropout and Academic Success** (UCI id 697)
Modelos: **KNN, SVM, Random Forest e XGBoost**
Entrega do artigo: **19/11 às 23h59** (AVA, um integrante envia)

---

## 0. Decisões tomadas

- **Notebook único** (`trabalho.ipynb`), com o fluxo EDA → pré-processamento → divisão → modelos → comparação. `knn.ipynb` e `svm.ipynb` são rascunhos de aprendizado.
- **Protocolo de avaliação:** hold-out estratificado de 20% para teste e `StratifiedKFold(5)` no treino (80%) para escolher os hiperparâmetros. As métricas finais e as matrizes de confusão são calculadas no conjunto de teste.
- **Mesmas partições para todos os modelos:** um único objeto `cv` e `RANDOM_STATE = 42`.
- **Desbalanceamento:** `class_weight='balanced'` (ou `sample_weight` no XGBoost) entra como configuração no grid. O critério de seleção é `f1_macro`.
- **Pré-processamento dentro do `Pipeline`**, para não vazar informação entre os folds.

---

## 1. Estrutura / setup

- [ ] **1.1** Criar `trabalho.ipynb` com seções em Markdown: EDA, Pré-processamento, Divisão dos dados, KNN, SVM, Random Forest, XGBoost, Comparação.
- [ ] **1.2** Concentrar todos os imports em uma célula no topo e remover os vários `!pip3 install` espalhados (as dependências vão para o `requirements.txt`, ver 10.1).
- [ ] **1.3** Definir `RANDOM_STATE = 42` e usá-lo em todos os pontos com aleatoriedade (split, CV, RF, XGBoost).
- [ ] **1.4** Carregar a base via `ucimlrepo.fetch_ucirepo(id=697)`, reaproveitando a célula já existente nos notebooks.

## 2. Análise exploratória (Especificação 3.1)

- [ ] **2.1** Informar o número de registros e de atributos e os tipos de cada um (`X.info()`, a tabela `variables` do ucimlrepo).
- [ ] **2.2** Identificar a variável-alvo `Target` (Dropout / Enrolled / Graduate) e descrever o significado de cada classe.
- [ ] **2.3** Verificar valores ausentes (`isna().sum()`) e registros duplicados (`duplicated().sum()`).
- [ ] **2.4** Procurar inconsistências: notas fora do intervalo esperado, unidades curriculares aprovadas > matriculadas, idades atípicas, códigos de categoria não documentados.
- [ ] **2.5** Analisar a distribuição do alvo (contagem + %, gráfico de barras) e comentar o desbalanceamento (Graduate ≈ 50%, Dropout ≈ 32%, Enrolled ≈ 18%).
- [ ] **2.6** *(Opcional)* Fazer gráficos de alguns atributos relevantes versus o alvo (ex.: notas do 1º/2º semestre, mensalidades em dia, bolsista) para usar no artigo.

## 3. Pré-processamento (Especificação 3.2)

- [ ] **3.1** Tratar os ausentes e as inconsistências encontrados em 2.3–2.4. Se não houver nenhum, registrar isso explicitamente no notebook e no artigo.
- [ ] **3.2** Classificar as colunas em:
  - nominais codificadas como inteiro (Marital status, Application mode, Course, Previous qualification, Nationality, Mother's/Father's qualification, Mother's/Father's occupation, Daytime/evening attendance…);
  - binárias (Displaced, Debtor, Gender, Scholarship holder, International…);
  - numéricas (notas, idade, unidades curriculares, taxa de desemprego, inflação, PIB…).
- [ ] **3.3** Aplicar one-hot nas nominais via `ColumnTransformer` (`OneHotEncoder(handle_unknown='ignore', min_frequency=...)`), agrupando as categorias raras para limitar a dimensionalidade.
- [ ] **3.4** Normalizar as numéricas com `StandardScaler` **dentro do Pipeline**.
- [ ] **3.5** Verificar e remover identificadores ou atributos não preditivos, e justificar a decisão no artigo (mesmo que nenhum seja removido).
- [ ] **3.6** Transformar o alvo em 1D com `y = y.squeeze()` (corrige o `DataConversionWarning` do KNN). Para o XGBoost, aplicar `LabelEncoder` no alvo.
- [ ] **3.7** Encapsular tudo em uma função ou objeto `preprocessador` reutilizado pelos 4 modelos.

## 4. Divisão dos dados e utilitários (Especificação 3.5 / 3.6)

- [ ] **4.1** Fazer o split `train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)`, reaproveitando `svm.ipynb`.
- [ ] **4.2** Criar **um único** `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)` e usá-lo em todos os `GridSearchCV`.
- [ ] **4.3** Criar a função `avaliar(modelo, nome)` que, no conjunto de teste, calcula acurácia, precisão, recall e F1 (macro e por classe), plota a matriz de confusão e devolve um dicionário com as métricas.
- [ ] **4.4** Usar `GridSearchCV` com `scoring` em várias métricas (`accuracy`, `precision_macro`, `recall_macro`, `f1_macro`) e `refit='f1_macro'`, para guardar todas as métricas da CV.

## 5. Ajustes no KNN (a partir de `knn.ipynb`)

- [ ] **5.1** Trocar o split atual (sem `stratify` e sem `random_state`) pelo split comum de 4.1.
- [ ] **5.2** Trocar o loop manual por `Pipeline(preprocessador, KNeighborsClassifier)` + `GridSearchCV` com o `cv` compartilhado:
  - `n_neighbors`: 3, 5, 7, 9, 11, 15…
  - `weights`: `uniform`, `distance`
  - `metric`: `euclidean`, `manhattan`
- [ ] **5.3** Corrigir os problemas atuais: o `classification_report` só mostra o último `k` do loop; o scaler está fora do pipeline; o `y` está em formato de coluna.
- [ ] **5.4** Refazer o gráfico "métrica × k" a partir do `cv_results_` (F1 macro médio na CV, uma curva por `weights`/`metric`).
- [ ] **5.5** Avaliar o melhor modelo com `avaliar()`.

## 6. Ajustes no SVM (a partir de `svm.ipynb`)

- [ ] **6.1** Trocar o `StandardScaler` isolado pelo `preprocessador` comum (com one-hot).
- [ ] **6.2** Adicionar `classifier__class_weight: [None, 'balanced']` ao grid (manter linear/rbf, `C` e `gamma`).
- [ ] **6.3** Usar o `cv` e o split comuns e avaliar com `avaliar()`.
- [ ] **6.4** Guardar o `cv_results_` em um DataFrame para a tabela de configurações (ver 9.2).

## 7. Random Forest (novo)

- [ ] **7.1** Montar `Pipeline(preprocessador, RandomForestClassifier(random_state=RANDOM_STATE))` + `GridSearchCV` com o `cv` comum:
  - `n_estimators`: 100, 300
  - `max_depth`: None, 10, 20
  - `min_samples_leaf`: 1, 5
  - `class_weight`: None, `balanced`
- [ ] **7.2** Avaliar o melhor modelo com `avaliar()`.
- [ ] **7.3** Fazer um gráfico com as importâncias dos atributos (top 15) para a discussão no artigo.

## 8. XGBoost (novo)

- [ ] **8.1** Adicionar `xgboost` às dependências e treinar com o alvo codificado (3.6).
- [ ] **8.2** Montar `Pipeline(preprocessador, XGBClassifier(objective='multi:softprob', random_state=RANDOM_STATE))` + `GridSearchCV` com o `cv` comum:
  - `n_estimators`: 100, 300
  - `max_depth`: 3, 6
  - `learning_rate`: 0.05, 0.1
- [ ] **8.3** Testar o tratamento do desbalanceamento com `sample_weight = compute_sample_weight('balanced', y_train)` passado no `fit`, comparando com e sem pesos.
- [ ] **8.4** Decodificar as predições para os rótulos originais e avaliar com `avaliar()`.

## 9. Comparação e resultados (Especificação 3.6)

- [ ] **9.1** Montar a tabela consolidada com a melhor configuração de cada modelo: F1 macro médio ± desvio na CV e acurácia, precisão, recall e F1 no teste.
- [ ] **9.2** Montar a tabela com todas as configurações testadas (pelo menos 2 por modelo) e suas métricas na CV.
- [ ] **9.3** Montar uma figura com as 4 matrizes de confusão lado a lado e um gráfico de barras comparando as métricas entre os modelos.
- [ ] **9.4** Exportar as figuras (`figuras/*.png`) e as tabelas (`resultados/*.csv`) para usar no artigo.
- [ ] **9.5** Escrever a interpretação dos resultados no notebook: melhor modelo, efeito do `class_weight`, desempenho na classe **Enrolled** (a mais difícil) e principais confusões entre classes.

## 10. Repositório / README

- [ ] **10.1** Criar o `requirements.txt` (ucimlrepo, pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost).
- [ ] **10.2** Criar o `README.md` com o objetivo, a base (link UCI), como executar e a estrutura do repositório.
- [ ] **10.3** Depois da migração para `trabalho.ipynb`, remover `knn.ipynb`/`svm.ipynb` (ou movê-los para `rascunhos/`).
- [ ] **10.4** Limpar os outputs ruidosos (logs de `pip install`, warnings).
- [ ] **10.5** Executar o notebook do zero ("Restart & Run All"), conferir se os resultados se reproduzem e confirmar que batem com o que está no artigo.
- [ ] **10.6** Deixar o repositório público (ou acessível ao professor) e pegar o link para o artigo.

## 11. Artigo IEEE (≤ 6 páginas, PDF, modelo IEEE Transactions do AVA)

- [ ] **11.1** Título e identificação dos autores.
- [ ] **11.2** Resumo: problema, objetivo, metodologia, principais resultados e conclusões.
- [ ] **11.3** Introdução: contexto de negócio (evasão no ensino superior), justificativa e objetivos.
- [ ] **11.4** Referencial teórico: KNN, SVM, Random Forest e XGBoost, com referências bibliográficas.
- [ ] **11.5** Metodologia: base, EDA, pré-processamento, divisão dos dados (hold-out + 5-fold), grids de hiperparâmetros e métricas.
- [ ] **11.6** Resultados: tabelas (9.1/9.2), matrizes de confusão e gráficos, com a discussão dos modelos e das configurações.
- [ ] **11.7** Conclusão: principais achados e trabalhos futuros.
- [ ] **11.8** Declaração do uso de IA (ferramentas e finalidade).
- [ ] **11.9** Referências (base UCI, artigo original da base, scikit-learn, XGBoost, referências dos algoritmos).
- [ ] **11.10** Incluir o link do GitHub no artigo.
- [ ] **11.11** Revisão final: limite de 6 páginas, texto autoral (sem plágio), figuras de terceiros referenciadas, coerência entre o artigo e o notebook, e qualidade do texto.
- [ ] **11.12** Exportar em PDF e enviar no AVA até **19/11 às 23h59** (um integrante).
