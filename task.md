# Tarefas — Trabalho Prático de Mineração de Dados (Grupo 7)

Base: **Predict Students' Dropout and Academic Success** (UCI id 697)
Modelos: **KNN, SVM, Random Forest e XGBoost**
Entrega do artigo: **19/11 às 23h59** (AVA, um integrante envia)

> As tarefas estão **na ordem de execução**. Cada tarefa depende das anteriores, exceto onde indicado:
> - As seções **5, 6, 7 e 8** (modelos) podem ser feitas em paralelo, uma por integrante, depois que a seção 4 estiver pronta.
> - As tarefas **11.1, 11.2 e 11.3** do artigo não dependem do código e podem começar a qualquer momento.

---

## 0. Decisões tomadas

- **Notebook único** (`trabalho.ipynb`), com o fluxo EDA → pré-processamento → divisão → modelos → comparação. `knn.ipynb` e `svm.ipynb` são rascunhos de aprendizado.
- **Protocolo de avaliação:** hold-out estratificado de 20% para teste e `StratifiedKFold(5)` no treino (80%) para escolher os hiperparâmetros. As métricas finais e as matrizes de confusão são calculadas no conjunto de teste.
- **Mesmas partições para todos os modelos:** um único objeto `cv` e `RANDOM_STATE = 42`.
- **Desbalanceamento:** `class_weight='balanced'` (ou `sample_weight` no XGBoost) entra como configuração no grid. O critério de seleção é `f1_macro`.
- **Pré-processamento dentro do `Pipeline`**, para não vazar informação entre os folds.

---

## 1. Setup

- [ ] **1.1** Criar o `requirements.txt` (ucimlrepo, pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost) e cada integrante instalar com `pip install -r requirements.txt`.
- [ ] **1.2** Criar `trabalho.ipynb` com as seções em Markdown: EDA, Pré-processamento, Divisão dos dados, KNN, SVM, Random Forest, XGBoost, Comparação.
- [ ] **1.3** Colocar todos os imports em uma célula no topo, sem `!pip3 install` no notebook.
- [ ] **1.4** Definir `RANDOM_STATE = 42` e usá-lo em todos os pontos com aleatoriedade (split, CV, RF, XGBoost).
- [ ] **1.5** Carregar a base via `ucimlrepo.fetch_ucirepo(id=697)`, reaproveitando a célula dos notebooks atuais.
- [ ] **1.6** Transformar o alvo em 1D logo após o carregamento: `y = y.squeeze()` (corrige o `DataConversionWarning` do KNN).

## 2. Análise exploratória (Especificação 3.1)

- [ ] **2.1** Informar o número de registros e de atributos e os tipos de cada um (`X.info()`, a tabela `variables` do ucimlrepo).
- [ ] **2.2** Identificar a variável-alvo `Target` (Dropout / Enrolled / Graduate) e descrever o significado de cada classe.
- [ ] **2.3** Verificar valores ausentes (`isna().sum()`) e registros duplicados (`duplicated().sum()`).
- [ ] **2.4** Procurar inconsistências: notas fora do intervalo esperado, unidades curriculares aprovadas > matriculadas, idades atípicas, códigos de categoria não documentados.
- [ ] **2.5** Analisar a distribuição do alvo (contagem + %, gráfico de barras) e comentar o desbalanceamento (Graduate ≈ 50%, Dropout ≈ 32%, Enrolled ≈ 18%).
- [ ] **2.6** *(Opcional)* Fazer gráficos de alguns atributos relevantes versus o alvo (ex.: notas do 1º/2º semestre, mensalidades em dia, bolsista) para usar no artigo.

## 3. Pré-processamento (Especificação 3.2)

- [ ] **3.1** Tratar os ausentes, duplicados e inconsistências encontrados em 2.3–2.4. Se não houver nenhum, registrar isso explicitamente no notebook e no artigo.
- [ ] **3.2** Verificar e remover identificadores ou atributos não preditivos, e justificar a decisão no artigo (mesmo que nenhum seja removido).
- [ ] **3.3** Classificar as colunas restantes em:
  - nominais codificadas como inteiro (Marital status, Application mode, Course, Previous qualification, Nationality, Mother's/Father's qualification, Mother's/Father's occupation, Daytime/evening attendance…);
  - binárias (Displaced, Debtor, Gender, Scholarship holder, International…);
  - numéricas (notas, idade, unidades curriculares, taxa de desemprego, inflação, PIB…).
- [ ] **3.4** Codificar o alvo com `LabelEncoder` (uma vez, para todos os modelos, porque o XGBoost exige rótulos numéricos). Guardar o `le` e usar `le.classes_` como nomes das classes nos relatórios e nas matrizes de confusão.
- [ ] **3.5** Montar o `preprocessador` com `ColumnTransformer`:
  - nominais → `OneHotEncoder(handle_unknown='ignore', min_frequency=...)`, agrupando as categorias raras;
  - numéricas → `StandardScaler`;
  - binárias → `passthrough`.

  Ele só é ajustado dentro do `Pipeline` de cada modelo, nunca na base inteira.

## 4. Divisão dos dados e utilitários (Especificação 3.5 / 3.6)

- [ ] **4.1** Fazer o split `train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)`, reaproveitando `svm.ipynb`.
- [ ] **4.2** Criar **um único** `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)`, que será usado em todos os `GridSearchCV`.
- [ ] **4.3** Definir o padrão de busca para todos os modelos: `GridSearchCV(pipeline, grid, cv=cv, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], refit='f1_macro', n_jobs=-1)`.
- [ ] **4.4** Criar a função `avaliar(modelo, nome)` que, no conjunto de teste, calcula acurácia, precisão, recall e F1 (macro e por classe), plota a matriz de confusão com `le.classes_` e devolve um dicionário com as métricas.
- [ ] **4.5** Criar a função `tabela_cv(grid_search, nome)` que transforma o `cv_results_` em um DataFrame (parâmetros + média ± desvio de cada métrica), para usar na seção 9.

## 5. KNN (migrar `knn.ipynb`) — paralelo com 6, 7 e 8

- [ ] **5.1** Montar `Pipeline(preprocessador, KNeighborsClassifier)` + `GridSearchCV` (padrão de 4.3), substituindo o loop manual e o split antigo:
  - `n_neighbors`: 3, 5, 7, 9, 11, 15…
  - `weights`: `uniform`, `distance`
  - `metric`: `euclidean`, `manhattan`
- [ ] **5.2** Fazer o gráfico "F1 macro × k" a partir do `cv_results_` (uma curva por `weights`/`metric`).
- [ ] **5.3** Avaliar o melhor modelo com `avaliar()` e gerar a tabela com `tabela_cv()`.

## 6. SVM (migrar `svm.ipynb`) — paralelo com 5, 7 e 8

- [ ] **6.1** Montar `Pipeline(preprocessador, SVC)` + `GridSearchCV` (padrão de 4.3), substituindo o `StandardScaler` isolado:
  - `kernel`: `linear`, `rbf`
  - `C`: 0.1, 1, 10
  - `gamma` (só rbf): `scale`, `auto`
  - `class_weight`: None, `balanced`
- [ ] **6.2** Avaliar o melhor modelo com `avaliar()` e gerar a tabela com `tabela_cv()`.

## 7. Random Forest (novo) — paralelo com 5, 6 e 8

- [ ] **7.1** Montar `Pipeline(preprocessador, RandomForestClassifier(random_state=RANDOM_STATE))` + `GridSearchCV` (padrão de 4.3):
  - `n_estimators`: 100, 300
  - `max_depth`: None, 10, 20
  - `min_samples_leaf`: 1, 5
  - `class_weight`: None, `balanced`
- [ ] **7.2** Avaliar o melhor modelo com `avaliar()` e gerar a tabela com `tabela_cv()`.
- [ ] **7.3** Fazer um gráfico com as importâncias dos atributos (top 15) para a discussão no artigo.

## 8. XGBoost (novo) — paralelo com 5, 6 e 7

- [ ] **8.1** Montar `Pipeline(preprocessador, XGBClassifier(objective='multi:softprob', random_state=RANDOM_STATE))` + `GridSearchCV` (padrão de 4.3):
  - `n_estimators`: 100, 300
  - `max_depth`: 3, 6
  - `learning_rate`: 0.05, 0.1
- [ ] **8.2** Rodar a busca duas vezes: sem pesos e com `fit(..., classifier__sample_weight=compute_sample_weight('balanced', y_train))`, para comparar o efeito do balanceamento.
- [ ] **8.3** Avaliar o melhor modelo com `avaliar()` e gerar as tabelas com `tabela_cv()`.

## 9. Comparação e resultados (Especificação 3.6) — depende de 5, 6, 7 e 8

- [ ] **9.1** Montar a tabela consolidada com a melhor configuração de cada modelo: F1 macro médio ± desvio na CV e acurácia, precisão, recall e F1 no teste.
- [ ] **9.2** Montar a tabela com todas as configurações testadas (pelo menos 2 por modelo) e suas métricas na CV, juntando as saídas de `tabela_cv()`.
- [ ] **9.3** Montar uma figura com as 4 matrizes de confusão lado a lado e um gráfico de barras comparando as métricas entre os modelos.
- [ ] **9.4** Escrever a interpretação dos resultados no notebook: melhor modelo, efeito do `class_weight`/`sample_weight`, desempenho na classe **Enrolled** (a mais difícil) e principais confusões entre classes.

## 10. Fechamento do notebook e do repositório

- [ ] **10.1** Executar o notebook do zero ("Restart & Run All") e confirmar que os resultados se reproduzem. **Estes números são os que vão para o artigo.**
- [ ] **10.2** Exportar as figuras (`figuras/*.png`) e as tabelas (`resultados/*.csv`) geradas nessa execução.
- [ ] **10.3** Remover `knn.ipynb`/`svm.ipynb` (ou movê-los para `rascunhos/`) e limpar os outputs ruidosos (warnings, logs).
- [ ] **10.4** Criar o `README.md` com o objetivo, a base (link UCI), como executar e a estrutura do repositório.
- [ ] **10.5** Fazer o commit/push final e deixar o repositório público (ou acessível ao professor).

## 11. Artigo IEEE (≤ 6 páginas, PDF, modelo IEEE Transactions do AVA)

- [ ] **11.1** Título e identificação dos autores. *(pode começar a qualquer momento)*
- [ ] **11.2** Introdução: contexto de negócio (evasão no ensino superior), justificativa e objetivos. *(pode começar a qualquer momento)*
- [ ] **11.3** Referencial teórico: KNN, SVM, Random Forest e XGBoost, com referências bibliográficas. *(pode começar a qualquer momento)*
- [ ] **11.4** Metodologia: base, EDA, pré-processamento, divisão dos dados (hold-out + 5-fold), grids de hiperparâmetros e métricas. *(depende de 1–4)*
- [ ] **11.5** Resultados: tabelas, matrizes de confusão e gráficos de 10.2, com a discussão dos modelos e das configurações. *(depende de 10.1)*
- [ ] **11.6** Conclusão: principais achados e trabalhos futuros.
- [ ] **11.7** Resumo: problema, objetivo, metodologia, principais resultados e conclusões (escrito por último, porque resume o artigo).
- [ ] **11.8** Declaração do uso de IA (ferramentas e finalidade).
- [ ] **11.9** Referências: base UCI, artigo original da base, scikit-learn, XGBoost e referências dos algoritmos. Conferir se todas as citações do texto estão na lista.
- [ ] **11.10** Incluir o link do GitHub (de 10.5) no artigo.
- [ ] **11.11** Revisão final: limite de 6 páginas, texto autoral (sem plágio), figuras de terceiros referenciadas, números do artigo iguais aos do notebook, qualidade do texto.
- [ ] **11.12** Exportar em PDF e enviar no AVA até **19/11 às 23h59** (um integrante).
