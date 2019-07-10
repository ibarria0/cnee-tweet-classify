import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.layers import Conv1D, Flatten
from tensorflow.keras.preprocessing import text
from tensorflow.keras import utils

from sklearn import preprocessing
from time import sleep
le = preprocessing.LabelEncoder()
import preprocessor as p

import csv


with open('tweets.csv') as csvfile:
    tweets = list(csv.reader(csvfile))
with open('train_data.csv') as csvfile:
    data = list(csv.reader(csvfile))[1:]

# load the pre-trained word-embedding vectors 
embeddings_index = {}
for i, line in enumerate(open('glove-sbwc.i25.vec')):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

X_train, Y_train = zip(*data)
X_train = [p.clean(x) for x in X_train]

#change to categorical
le.fit(Y_train)
Y_train = utils.to_categorical(le.transform(Y_train), 2)

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=70)

# create token-embedding mapping
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#####CNN MODEL
input_layer = layers.Input((70, ))
embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)
pooling_layer = layers.GlobalMaxPool1D()(conv_layer)
output_layer1 = layers.Dense(512, activation="relu")(pooling_layer)
output_layer1 = layers.Dropout(0.25)(output_layer1)
output_layer2 = layers.Dense(2, activation="sigmoid")(output_layer1)
model = models.Model(inputs=input_layer, outputs=output_layer2)
model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10)

for t in tweets:
    tweet = p.clean(t[3])
    text = sequence.pad_sequences(tokenizer.texts_to_sequences([tweet]), maxlen=70)
    prediction = model.predict(text)
    if prediction[0][0] > 0.8:
        print(prediction[0][0], tweet)
