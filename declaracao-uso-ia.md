# Declaração do Uso de IA

**Trabalho:** Predição de Evasão e Sucesso Acadêmico — Grupo 7
**Disciplina:** BSI — Inteligência de Negócios — 2026-2

## Ferramentas utilizadas

Utilizamos o **Claude Code** (modelo Claude Opus 5.5, da Anthropic) no desenvolvimento do código e na organização do repositório.

## O que fizemos sem IA

As primeiras versões do KNN e do SVM, que estão na pasta `rascunhos/`, foram feitas por nós como exercício, antes de usarmos a IA. As decisões sobre como conduzir os experimentos também foram nossas: usar um único notebook, separar 20% dos dados para teste, usar validação cruzada com 5 partições no treino e tratar o desbalanceamento das classes usando o F1 macro como critério de escolha dos modelos.

## Como usamos a IA no código

**Planejamento.** Usamos o Claude Code para ler a especificação e os nossos rascunhos e montar a lista de tarefas (`tasks.md`). A ferramenta também apontou o que precisava mudar no KNN e no SVM para atender à especificação e organizou as tarefas na ordem em que precisavam ser feitas.

**Implementação.** A partir das nossas decisões, a ferramenta escreveu o código do notebook principal (`trabalho.ipynb`): a análise exploratória, o pré-processamento, o treinamento dos quatro classificadores (KNN, SVM, Random Forest e XGBoost) com busca de hiperparâmetros e a comparação dos resultados. Os valores testados para os hiperparâmetros foram sugeridos pela IA. Os gráficos e as tabelas usados no artigo também foram gerados por esse código.

**Execução e conferência.** A ferramenta executou o notebook, corrigiu problemas que apareceram nos gráficos e nas tabelas e executou tudo de novo do zero para conferir se os resultados se repetiam. As duas execuções deram os mesmos resultados.

**Textos e documentação.** Os textos explicativos dentro do notebook, incluindo a interpretação dos resultados, foram escritos pela IA com base nos resultados obtidos. A ferramenta também criou o `README.md`, o `requirements.txt` e organizou os arquivos do repositório.

## Revisão

Revisamos o conteúdo produzido com ajuda da IA e assumimos a responsabilidade por ele. Os resultados podem ser conferidos executando o `trabalho.ipynb`, seguindo as instruções do `README.md`.