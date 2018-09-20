# Pokémon
### Trabalho da Udacity Brasil

**Aluno:** Paulo Vitor Pereira Cotta

A proposta do trabalho é executar processos onde os pokémons possam batalhar entre rodadas, sendo que cada treinador possa escolher apenas um pokémon.

A ideia é saber em que cada rodada qual Pokémon possa vencer.

O DataSet escolhido foi retirado da plataforma Kaggle, conforme link a seguir: https://www.kaggle.com/terminus7/pokemon-challenge/kernels.

**Segue uma introdução sobre o local que será utilizado:
Kanto (em japonês: カントー地方, Kantō-chihō) é um país ou região fictícia, da série Pokémon. Sua geografia é baseada na região de Kanto, uma região da ilha de Honshu, no Japão, de onde vem seu nome. A semelhança entre as formações de baía vistas no mapa do jogo e as formações reais de Sagami Bay, Suruga Bay e a Baía de Tokyo é particularmente impressionante.
Kanto localiza-se a leste de Johto; presumivelmente, eles formam um pequeno continente.
Fonte: Wikipédia: https://pt.wikipedia.org/wiki/Kanto_(Pok%C3%A9mon)**


Imagem


Agora o problema em nossa mão é que, dado alguns recursos sobre cada Pokemon, como seu ataque, defesa ou seu valor de velocidade, etc, precisamos prever o vencedor de uma batalha Pokemon aleatória que nunca ocorreu antes.

Um conjunto de dados é a coleta de grande quantidade de dados sobre um tópico específico. Este conjunto de dados nos fornece informações como os Pontos de Vida, ataque, Defesa Especial e tempo em que o Pokémon é lendário **TRUE** ou não **FALSE**. A tabela acima mostra os dados dos primeiros 5 Pokemon, mas há um total de 800 Pokemon (significa 800 linhas) no conjunto de dados.

## Decision Trees

![alt text](http://i.imgur.com/FDwpwFJ.jpg "Árvore Pokémon")

Suponha que temos uma tarefa de prever o animal com base nas características como tipo, altura, peso ou velocidade do animal. Esta tarefa pode ser facilmente modelada usando uma árvore de decisão. Então, em cada ponto da árvore de decisão, fazemos uma pergunta e, dependendo da resposta, dividimos a árvore em sub-árvores. Este processo é repetido até prevermos um animal. Assim, dado um conjunto de dados, um Classificador de Árvore de Decisão fará as perguntas certas (aumentando o Ganho de Informações) em cada ponto, de modo a dividir a árvore de forma a aumentar a confiança para cada previsão (aumentando a pureza do resultado).

Como as florestas são uma coleção de árvores, o Random Forest Classifier usa várias árvores de decisão e, finalmente, combina os resultados de cada árvore de decisão para prever seu resultado final.

Por fim, foi criado um classificador de floresta aleatório da seguinte maneira. Os n_estimators fornecem o Número de árvores de decisão valor=100 usados para fazer a floresta.


#### Floresta com 100 árvores de decisão

Reiterando a tarefa, dois Pokemon com seu conjunto de recursos (velocidade, ataque, etc) qual deles vai ganhar.

O treinamento do classificador no conjunto de dados de Pokémon (ou seja, x_train) e minimizamos a perda entre os valores previstos e reais (y_train) no conjunto de treinamento. Treinar aqui significa encontrar as relações entre os diferentes recursos do conjunto de dados para fazer previsões.

Em seguida, calculamos a precisão de nosso classificador, que é de 65% (significa que nosso classificador irá prever resultados corretos para 95 de 100 partidas), o que é uma boa precisão para começar.

Até agora, concluímos todas as etapas necessárias desde a criação de um classificador até o treinamento e agora será testado efetuando uma batalha Pokémon.

Imagem


#### Processo explicativo da predição

As duas colunas correspondem aos Pokémon que vão competir. Será alimentado esses dois Pokemon no classificador e ele retornará o vencedor mais provável para essa batalha. Lembre-se de que o classificador não está apenas predizendo aleatoriamente o vencedor. Na verdade, ele está analisando vários parâmetros cuidadosamente para chegar à decisão correta.


Imagem


## Conclusão

Foi visto que um problema bem básico que pode ser resolvido usando o Machine Learning. Os conceitos abordados neste trabalho formam a base da maioria das abordagens de Machine Learning. A explicação dos conceitos da maneira mais simples, para que possa ter uma compreensão justa de como o Machine Learning funciona e pode ser usado no mundo real.