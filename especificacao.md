BSI - Inteligência de Negócios - 2026-2 Trabalho prático de Mineração de Dados 
1. Descrição do trabalho 
O objetivo deste trabalho é aplicar e comparar algoritmos de classificação supervisionada em bases de dados públicas. Os grupos deverão realizar uma análise exploratória dos dados, implementar classificadores, executar experimentos com diferentes configurações de hiperparâmetros e avaliar os resultados utilizando as métricas solicitadas.

2. Bases de dados 
Será utilizada a seguinte base pública: 
Predict Students' Dropout and Academic Success: classificação da situação acadêmica de estudantes, considerando as categorias evasão, matrícula e conclusão do curso. Disponível em https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success.

3. Requisitos 
3.1. Análise exploratória dos dados 
Deverá contemplar asseguintes tarefas: 
• Identificar o número de registros, os atributos e seus respectivos tipos. 
• Identificar a variável-alvo. 
• Verificar a existência de valores ausentes, duplicados e possíveis inconsistências.
• Analisar a distribuição da variável-alvo, identificando possíveis desbalanceamentos.

3.2. Pré-processamento 
Deverão ser aplicadas as etapas de pré-processamento necessárias à base utilizada, considerando: 
• Tratamento de valores ausentes e inconsistências. 
• Codificação de atributos categóricos. 
• Normalização dos atributos numéricos. 
• Remoção de identificadores que não sejam apropriados como atributos preditivos.

3.3. Classificadores obrigatórios 
O grupo deverá implementar os seguintes algoritmos: 
• K-Nearest Neighbors (KNN); 
• Support Vector Machine (SVM).

3.4. Experimentos específicos por grupo 
Além dos requisitos comuns, cada grupo deverá realizar os experimentos adicionais definidos para sua base. 

Predict Students' Dropout and Academic Success 
Grupo 7: Implementar KNN, SVM, Random Forest e XGBoost.

3.5. Validação cruzada e ajuste de hiperparâmetros 
Os grupos deverão utilizar validação cruzada estratificada com cinco partições (5-fold cross-validation) para avaliar diferentes configurações dos classificadores (pelo menos 2 de cada). Utilize a mesma divisão dos dados em partições para todos os classificadores. 

3.6. Métricas de avaliação 
Os grupos deverão apresentar e interpretar: 
• Acurácia (accuracy); 
• Precisão (precision); 
• Revocação (recall); 
• F1-score; 
• Matriz de confusão. 

4. Artigo científico 
Os resultados do trabalho deverão ser apresentados na forma de um artigo científico, redigido em linguagem acadêmica e seguindo o modelo do IEEE transaction, disponibilizado no AVA, ser entregue no formato PDF, e ter no máximo 6 páginas (incluindo imagens, tabelas e referências). O artigo deverá conter, no mínimo, as seguintes seções: 

1. Título e identificação dos autores: título que represente o problema investigado, identificação dos integrantes do grupo e demais informações solicitadas.
2.  Resumo: síntese dos seguintes elementos: problema, objetivo, metodologia, principais resultados e conclusões.
3.  Introdução: contextualização do problema de negócio, justificativa e apresentação dos objetivos.
4.  Referencial teórico: fundamentação sobre os algoritmos utilizados, apoiada em referências bibliográficas.
5.  Metodologia: descrição da base de dados, do pré-processamento, da divisão dos dados, dos experimentos e das métricas de avaliação.
6.  Resultados: apresentação dos resultados (por meio de tabelas, gráficos e matrizes de confusão), acompanhada de uma discussão sobre os modelos e as configurações investigadas.
7.  Conclusão: síntese dos principais achados e possibilidades de trabalhos futuros.
8.  Declaração do uso de IA: informar se houve o uso de ferramentas de IA e a finalidade do uso.
9.  Referências: lista das fontes citadas no artigo.

Observações importantes:
• Trabalhos que excederem o número de páginas serão penalizados.
• Consulte as fontes, referencie-as, mas escreva o texto com suas próprias palavras. Caso seja detectado o plágio, o trabalho será avaliado com nota zero.
• A utilização de figuras/tabelas ... que não foram produzidas pelos autores devem referenciar os trabalhos de onde foram retirados. 
• Documentos sem referências bibliográficas serão fortemente penalizados.
• Documentos desacompanhados do código fonte serão avaliados com nota zero. 

5. Implementação 
O grupo deverá colocar no artigo o link do Github contendo o notebook em Python utilizado nos experimentos, contendo o código necessário para reproduzir a análise exploratória, o pré-processamento, o treinamento dos classificadores, devendo estar coerente com os procedimentos descritos no artigo.

6. Apresentação oral do artigo 
Cada grupo deverá apresentar oralmente o artigo científico para a turma, utilizando slides contendo principais pontos do artigo. As apresentações ocorrerão nos 23, 26 e 30/11, com duração de 5 a 20 minutos. Os grupos associados a uma mesma base deverão apresentar no mesmo dia. É importante que os grupos não comparem ou discutam os resultados obtidos antes da apresentação oral. 

7. Outras informações 
• O artigo produzido deve ser entregue por meio do AVA até as 23h59 do dia 19/11 (somente um integrante do grupo deverá enviar)
• O trabalho deverá ser realizado em grupo de 3 a 4 pessoas sendo que o número máximo de grupos é 10.
• Nota máxima: 40 pontos
• Caso seja detectada algum tipo de fraude, o trabalho será avaliado com nota zero.
• Além do conteúdo técnico, será avaliada a qualidade do texto produzido, que deve ser claro, objetivo e ter as informações apresentadas de forma organizada.