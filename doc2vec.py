from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
import preprocessing



def doc2vec_Fun ():
    df = pd.read_excel('fake_new_dataset.xlsx')
    df.title = df.title.astype(str)
    df.text = df.text.astype(str)

    df['news'] = df['title'] + df['text']
    df.drop(labels=['title', 'text'], axis=1, inplace=True)
    df.drop(labels=['subcategory'], axis=1, inplace=True)
    list_label = [0,0,1,1,0]
    doc = []
    for item in df['news']:
        item = preprocessing.text_preprocessing(item)
        doc.append(item)
        if len(doc) == 5:
            break
    tokenized_doc = []
    for d in doc:
        tokenized_doc.append(word_tokenize(d.lower()))
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
    model=Doc2Vec(tagged_data,vector_size=100,window=2, min_count=1,workers=4,epochs=100)
    list_data = []
    for index in range(0,len(model.dv)):
        list_data.append(model.dv[index])
    return list_data,list_label

def get_data():
    df = pd.read_excel('fake_new_dataset.xlsx')
    df.title = df.title.astype(str)
    df.text = df.text.astype(str)

    df['news'] = df['title'] + df['text']
    df.drop(labels=['title', 'text'], axis=1, inplace=True)
    df.drop(labels=['subcategory'], axis=1, inplace=True)
    return df['news']