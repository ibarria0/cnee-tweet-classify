import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.layers import Conv1D, Flatten
from tensorflow.keras.preprocessing import text
from tensorflow.keras import utils
import matplotlib.pyplot as plt
from random import shuffle

from sklearn import preprocessing
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from time import sleep
import preprocessor as p
import csv
from plots import plot_confusion_matrix


with open('tweets.csv') as csvfile:
    tweets = list(csv.reader(csvfile))
with open('train_data4.csv', encoding = "ISO-8859-1") as csvfile:
    data = list(csv.reader(csvfile))[1:]

# load the pre-trained word-embedding vectors 
#embeddings_index = {}
#for i, line in enumerate(open('glove-sbwc.i25.vec')):
#    values = line.split()
#    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

X_train, Y_train = zip(*data)
Y_train = ['otros' if y == "otros" else 'educacion' for y in Y_train]
#print(Y_train)
X_train = [p.clean(x) for x in X_train]

####SEPARATE DATA
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X_train, Y_train)

# label encode the target variable
encoder = preprocessing.LabelEncoder()
encoder.fit(Y_train)
class_names = list(encoder.classes_)
print(class_names)
train_y = encoder.transform(train_y)
valid_y = encoder.transform(valid_y)

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(train_x)
word_index = tokenizer.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors
train_seq_x = sequence.pad_sequences(tokenizer.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(tokenizer.texts_to_sequences(valid_x), maxlen=70)

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    print('vocab coverage in embedding %f' % ((nonzero_elements / vocab_size)*100))
    return embedding_matrix

embedding_matrix = create_embedding_matrix('embeddings-new_large-general_3B_fasttext.vec',tokenizer.word_index, 300)

#####CNN MODEL
input_layer = layers.Input((70, ))
embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=True)(input_layer)
embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
conv_layer = layers.Convolution1D(64, 3, activation="relu")(embedding_layer)
pooling_layer = layers.GlobalMaxPool1D()(conv_layer)
output_layer1 = layers.Dense(128, activation="relu")(pooling_layer)
output_layer1 = layers.Dropout(0.25)(output_layer1)
output_layer2 = layers.Dense(2, activation="sigmoid")(output_layer1)
model = models.Model(inputs=input_layer, outputs=output_layer2)
model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_seq_x, train_y, epochs=30, validation_data=(valid_seq_x, valid_y))

loss, accuracy = model.evaluate(train_seq_x, train_y, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(valid_seq_x, valid_y, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# predict the labels on validation dataset
predictions = model.predict(valid_seq_x)
predictions = predictions.argmax(axis=-1)
print(metrics.accuracy_score(predictions, valid_y))
matrix = metrics.confusion_matrix(predictions, valid_y)
print(matrix)


#np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
#plot_confusion_matrix(valid_y, predictions, classes=class_names,
#                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
#plot_confusion_matrix(valid_y, predictions, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')

#plt.show()

#sleep(60)

shuffle(tweets)
for t in tweets[:20]:
    tweet = p.clean(t[3])
    text = sequence.pad_sequences(tokenizer.texts_to_sequences([tweet]), maxlen=70)
    prediction = model.predict(text)
    label = list(prediction[0]).index(max(list(prediction[0])))
    print(encoder.inverse_transform([label]), tweet)

while True:
  text_s = input("Escribe aqui un tweet de prueba: ")
  text = sequence.pad_sequences(tokenizer.texts_to_sequences([text_s]), maxlen=70)
  prediction = model.predict(text)
  print(prediction)
  label = list(prediction[0]).index(max(list(prediction[0])))
  print(encoder.inverse_transform([label]), text_s)

