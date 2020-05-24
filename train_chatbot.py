# -*- coding:utf-8 -*-
"""
1. Import de bibliotecas e carregamento de dados
"""
import json
import pickle
import random

import nltk
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# load intents json file
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

"""
2. Pré Processamento dos Dados

O modelo não pode receber os dados brutos. 
Ele precisa passar por pré-processamento para que a máquina possa entender facilmente. 
Para dados textuais, existem muitas técnicas de pré-processamento disponíveis. 
A primeira técnica é a tokenização, na qual dividimos as frases em palavras.  
Observando o arquivo de intenções (intents), podemos ver que cada tag contém uma lista de padrões e respostas. 
Nós simbolizamos cada padrão e adicionamos as palavras em uma lista. 
Além disso, criamos uma lista de classes e documentos para adicionar todas as intenções associadas aos padrões. 
"""
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # add documents in the corpus
        documents.append((word, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)

# lemmaztize and lower each word and remove duplicates
"""
Outra técnica é a Lematização. 
Podemos converter palavras na forma de lema para reduzir todas as palavras canônicas. 
Por exemplo, as palavras jogo, jogo, jogo, jogo, etc. serão todas substituídas por jogo. 
Dessa forma, podemos reduzir o número total de palavras em nosso vocabulário. 
Portanto, agora vamos lematizar cada palavra e remover as palavras duplicadas. 
"""
words = [lemmatizer.lemmatize(w.lower())
         for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

"""
No final, as palavras contêm o vocabulário do nosso projeto e as classes contêm o total de entidades a serem classificadas. 
Para salvar o objeto python em um arquivo, usamos o método pickle.dump(). 
Esses arquivos serão úteis após o término do treinamento e prevemos os bate-papos.
"""
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

"""
3. Crie dados de treinamento e teste

Para treinar o modelo, converteremos cada padrão de entrada em números. 
Primeiro, vamos lematizar cada palavra do padrão e criar uma lista de zeros do mesmo tamanho que o número total de palavras. 
Definiremos o valor 1 apenas para o índice que contém a palavra nos padrões. 
Da mesma maneira que criaremos a saída, definindo 1 para o padrão de entrada da classe ao qual pertence. 
"""
# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

"""
4. Treinando o modelo

A arquitetura do nosso modelo será uma rede neural composta por 3 camadas densas. 
A primeira camada possui 128 neurônios, 
a segunda possui 64 e 
a última camada possui os mesmos neurônios que o número de classes. 
As camadas de eliminação são introduzidas para reduzir a super adaptação do modelo. 
Usamos o otimizador SGD e ajustamos os dados para iniciar o treinamento do modelo. 
Após a conclusão do treinamento de 200 épocas, 
salvamos o modelo treinado usando a função Keras model.save("chatbot_model.h5"). 
"""
# Create model -
# 3 layers. First layer 128 neurons,
# second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")
