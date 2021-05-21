import numpy as np
import tensorflow as tf
import pandas as pd
import doc2vec
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import models,layers

# Importing libraries
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


list_data,list_label = doc2vec.doc2vec_Fun()


X_train, X_test, y_train, y_test = train_test_split(list_data,list_label, test_size=0.1, random_state=37)
model = Sequential()
#print(X_train)
#print(np.shape(X_train))
model.add(Embedding(100, 8, input_length=100))
model.add(Conv1D(16,4 ,padding='valid',activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(32,4,activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
#model.add(Dense(12, activation='relu', input_shape=(100,)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy() ,metrics=[tf.keras.metrics.BinaryAccuracy()])
#model.fit(x=X_train,y=y_train,batch_size=128,epochs=10,validation_data=(X_test,y_test))
#model.fit(x=np.array(X_train),y=np.array(y_train),batch_size=128,epochs=10,validation_data=(np.array(X_test),np.array(y_test)))
model.summary()
scores = model.evaluate(np.array(X_test), np.array(y_test), verbose=1)
print("Accuracy: %.2f%%" % (scores[1] * 100))