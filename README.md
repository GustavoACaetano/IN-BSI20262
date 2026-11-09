# Predição de Evasão e Sucesso Acadêmico — Grupo 7

Trabalho prático de Mineração de Dados da disciplina **Inteligência de Negócios (BSI — 2026-2)**.

O objetivo é classificar a situação acadêmica de estudantes do ensino superior em **evasão (Dropout)**, **matrícula ativa (Enrolled)** ou **conclusão (Graduate)**, comparando quatro classificadores supervisionados: **KNN, SVM, Random Forest e XGBoost**.

## Base de dados

[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (UCI Machine Learning Repository, id 697): 4424 estudantes, 36 atributos (demográficos, socioeconômicos, macroeconômicos e de desempenho acadêmico nos dois primeiros semestres) e a variável-alvo `Target`.

O notebook baixa a base automaticamente pelo pacote `ucimlrepo`, então é necessário ter acesso à internet na primeira execução.

## Metodologia (resumo)

1. **Análise exploratória:** tipos de atributos, valores ausentes, duplicados, regras de consistência e distribuição do alvo (desbalanceado: ~50% / 32% / 18%).
2. **Pré-processamento** (dentro de um `Pipeline`, para não vazar informação entre as partições):
   - *one-hot encoding* dos atributos nominais, com agrupamento das categorias raras;
   - padronização dos atributos numéricos;
   - atributos binários mantidos como 0/1.
3. **Divisão:** 80% treino / 20% teste, estratificada (`random_state=42`).
4. **Ajuste de hiperparâmetros:** `GridSearchCV` com validação cruzada estratificada de 5 partições, **as mesmas para todos os modelos**, usando o F1 macro como critério de seleção. O balanceamento de classes (`class_weight` / `sample_weight`) é testado como configuração.
5. **Avaliação no teste:** acurácia, precisão, revocação, F1 (macro e por classe) e matriz de confusão.

## Como executar

Requisitos: Python 3.10 ou superior.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Depois, abra o `trabalho.ipynb` no Jupyter ou no VS Code e execute todas as células (*Restart & Run All*). Também é possível executar pela linha de comando:

```bash
jupyter nbconvert --to notebook --execute --inplace trabalho.ipynb
```

A execução completa leva alguns minutos, a maior parte na busca de hiperparâmetros do SVM. Os resultados são reprodutíveis porque todas as fontes de aleatoriedade usam `random_state=42`.

## Estrutura do repositório

| Caminho | Conteúdo |
|---|---|
| `trabalho.ipynb` | Notebook principal: EDA, pré-processamento, os 4 modelos e a comparação |
| `figuras/` | Figuras geradas pelo notebook (usadas no artigo) |
| `resultados/` | Tabelas em CSV: resultados da validação cruzada por modelo, resumo dos modelos, efeito do balanceamento, importância dos atributos |
| `rascunhos/` | Primeiras implementações de KNN e SVM, feitas para aprendizado (não fazem parte dos resultados) |
| `requirements.txt` | Dependências Python |
| `task.md` | Lista de tarefas do grupo |
| `especificacao.md` | Enunciado do trabalho |
