import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
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

X_train, Y_train = zip(*data)
X_train = [p.clean(x) for x in X_train]

tokenizer = text.Tokenizer(num_words=2000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_matrix(X_train)
#X_train = sequence.pad_sequences(X_train, maxlen=maxlen)

le.fit(Y_train)
Y_train = utils.to_categorical(le.transform(Y_train), 2)


model = Sequential()
model.add(Dense(512, input_shape=(2000,), activation=tf.nn.relu))
model.add(Dense(2, activation=tf.nn.softmax))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

trained = model.fit(X_train, Y_train, epochs=10)

for t in tweets:
    tweet = p.clean(t[3])
    text = tokenizer.texts_to_matrix([tweet])
    prediction = model.predict(text)
    if prediction[0][0] > 0.7:
        print(prediction[0][0], tweet)
