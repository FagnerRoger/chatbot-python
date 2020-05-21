# chatbot-python

Build Chatbot Project Using Python

#   Pré-requisitos
Para implementar o chatbot, usaremos o Keras, que é uma biblioteca de Deep Learning, o NLTK, que é um Natural Language Processing Toolkit e algumas bibliotecas úteis. Execute o comando abaixo para garantir que todas as bibliotecas estejam instaladas
```
pip install tensorflow keras pickle-mixin nltk
sudo apt-get install python3-tk
```
  
# Como os Chatbots funcionam?
Os chatbots nada mais são do que um software inteligente que pode interagir e se comunicar com pessoas como seres humanos. Interessante não é! Então agora vamos entender como eles realmente funcionam. Todo o chatbot é baseado nos conceitos de NLP (Natural Language Processing). 
A NLP é composta de duas coisas:
1.  NLU (Natural Language Understanding): A capacidade das máquinas de entender a linguagem humana como o inglês.
2.  NLG (Natural Language Generation): a capacidade de uma máquina de gerar texto semelhante a frases escritas por humanos.

Exemplo: Usuário fazendo uma pergunta para um chatbot: 
"Ei, quais as notícias hoje? 
O chatbot dividirá a sentença do usuário em duas coisas: 
Intenção e uma Entidade. 
A intenção dessa frase pode ser get_news, pois se refere a uma ação que o usuário deseja executar. 
A entidade informa detalhes específicos sobre a intenção, então aqui 'hoje' será a entidade. 
Portanto, é usado um modelo de aprendizado de máquina para reconhecer as intenções e entidades do bate-papo.

# Estrutura do arquivo do projeto
Vamos analisar rapidamente cada um deles, dando uma idéia de como o projeto está implementado.
1.  **train_chatbot.py -**   Neste arquivo, construímos e treinaremos o modelo de deep learning que pode classificar e identificar o que o usuário está pedindo ao bot.
2.  **gui_chatbot.py -**   É neste arquivo que criamos uma interface gráfica do usuário para conversar com nosso chatbot treinado.
3.  **intents.json -**   O arquivo de intenções possui todos os dados que usaremos para treinar o modelo. Ele contém uma coleção de tags com seus padrões e respostas correspondentes.
4.  **chatbot_model.h5 -**   Este é um arquivo de formato de dados hierárquico no qual armazenamos os pesos e a arquitetura do nosso modelo treinado.
5.  **classes.pkl -**    Este arquivo pode ser usado para armazenar todos os nomes de tags para classificar quando estamos prevendo a mensagem.
6.  **words.pkl -**    Este arquivo contém todas as palavras exclusivas que são o vocabulário do nosso modelo.

# Running the chatbot
```
python3 train_chatbot.py
```