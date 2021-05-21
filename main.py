import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import models,layers
# Importing libraries
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import  AveragePooling1D
from sklearn.metrics import accuracy_score, classification_report
from keras.preprocessing import sequence
from sklearn import metrics
from tensorflow.python.ops.confusion_matrix import confusion_matrix
import pickle
import preprocessing as preprocess


if __name__ == '__main__':
   df=pd.read_csv('covid19_articles.csv')
   df2=pd.read_csv('english_test_with_labels.csv')
   df3=pd.read_excel('fake_new_dataset.xlsx')
   df4=pd.read_csv('train_hash_pre_lemm.csv')
   df5=pd.read_csv('covid_clean_dataset_09_02_21 (1).csv')
   #print(df3['title'])
   # df.title = df.title.astype(str)
   # df.text = df.text.astype(str)
   a=0
   b=0
   for i in range(len(df['label'])):
      if df['label'][i]=='FAKE':
         df['label'][i]=0
         a=a+1
      else:
         df['label'][i]=1
         b=b+1
   print('number of fake is ',a)
   print('number of real is ',b)
   #print(df['label'])
   for i in range(len(df2['label'])):
      if df2['label'][i]=='fake':
         df2['label'][i]=0
      else:
         df2['label'][i]=1
   for i in range(len(df4['title'])):
      if df4['fake'][i]==1:
         df4['label'][i]=0
      else:
         df4['label'][i]=1

#   df3.drop(labels=['subcategory', 'text'], axis=1, inplace=True)
#   df4.drop(labels=['fake', 'real'], axis=1, inplace=True)


   df3.title=df3.title.astype(str)
   frames = [df,df3]
   finaldf = pd.concat(frames,axis=0, ignore_index=True)
   #finaldf.drop(labels=['id'], axis=1, inplace=True)

   for i in range(len(finaldf)):
      print(i,finaldf['title'][i])
   print(len(finaldf))
   # df1 = pd.read_csv('Fake.csv')
   # df2 = pd.read_csv('True.csv')
   # df2['target'] = 1
   # df1['target'] = 0
   # frames = [df1, df2]
   # df = pd.concat(frames)
   # df['news']=df['title']+df['text']
   # df.drop(labels=['title', 'text'], axis=1, inplace=True)
   # df.drop(labels=['subcategory'],axis=1,inplace=True)
   # df.drop(labels=['title', 'text'], axis=1, inplace=True)
   # df.drop(labels=['subject', 'date'], axis=1, inplace=True)
   finaldf = finaldf.sample(frac=1)
   X_train, X_test, y_train, y_test = train_test_split(df5.text, df5.target, test_size=0.3, random_state=37)
   tk = Tokenizer(num_words=10000,
                  filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n', lower=True, split=" ")
   tk.fit_on_texts(X_train)
   X_train_seq = tk.texts_to_sequences(X_train)
   X_test_seq = tk.texts_to_sequences(X_test)
   # max=0
   # maxtest=0
   # for item in X_train_seq:
   #    if(len(item)>max):
   #       max=len(item)
   # for item in X_test_seq:
   #    if(len(item)>maxtest):
   #       maxtest=len(item)

   X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=100)
   X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=100)
   model = Sequential()  # initilaizing the Sequential nature for CNN model
   print(len(tk.index_word))

   # Adding the embedding layer which will take in maximum of 450 words as input and provide a 32 dimensional output of those words which belong in the top_words dictionary
   model.add(Embedding(len(tk.index_word), 32, input_length=100))
   model.add(LSTM(100))
   # model.add(Conv1D(16, 4, padding='valid', activation='relu'))
   # model.add(MaxPooling1D())
   # model.add(Conv1D(32, 4, activation='relu'))
   # model.add(MaxPooling1D())
   # model.add(Flatten())
   #model.add(Dense(12, activation='relu', input_shape=(100,)))
   #model.add(Dense(128, activation='relu'))
   #model.add(Dense(250, activation='relu'))
   #model.add(Dense(32, activation='relu'))
   model.add(Dense(1, activation='sigmoid'))

   model.compile( loss='binary_crossentropy',optimizer='adam',
                 metrics=['accuracy'])
   #model.fit(x=X_train,y=y_train,batch_size=128,epochs=10,validation_data=(X_test,y_test))
   #model.fit(x=np.array(X_train_seq_trunc),y=np.array(y_train),batch_size=128,epochs=10,validation_data=(np.array(X_test_seq_trunc),np.array(y_test)))
   print(X_train_seq_trunc)
   print(np.array(y_train))
   X_train_array = np.asarray(X_train_seq_trunc, dtype=np.int)
   y_train_array = np.asarray(y_train, dtype=np.int)

   X_test_array = np.asarray(X_test_seq_trunc, dtype=np.int)
   y_test_array = np.asarray(y_test, dtype=np.int)
   model.fit(X_train_array, y_train_array,validation_data=(X_test_array,y_test_array), epochs=5, batch_size=64)
   model.save("myModel.h5")

   predications= model.predict(X_test_array)
   predications= predications>=0.5

   model.summary()
   scores = model.evaluate(X_test_array, y_test_array, verbose=0)
   print("Accuracy: %.2f%%" % (scores[1] * 100))
   print("Accuracy:", metrics.accuracy_score(y_test_array, predications))
   y_pred2 =model.predict_classes(X_test_array)
   print(confusion_matrix(y_test_array, y_pred2))


   # emb_model.fit(x=X_train_seq_trunc, y=y_train, batch_size=128, epochs=10, validation_data=(X_test_seq_trunc, y_test))
   doc = ["COVID does not exist"]

   tk.fit_on_texts(doc)
   test_text = tk.texts_to_sequences(doc)
   print(test_text)
   test_seq = pad_sequences(test_text, maxlen=100)
   predications = model.predict(test_seq)
   print(test_seq)
   print(predications)

   # print(predications)
   predications = (predications < 0.5)
   print(predications)


   # saving
   with open('tokenizer.pickle', 'wb') as handle:
      pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)



   # false=0
   # true=0
   # for item in predications:
   #    if item[0]>=0.5:
   #       true=true+1
   #    else:
   #       false=false+1
   #
   # if true>false:
   #    print('true')
   # else: print('false')

   #print(X_test)




