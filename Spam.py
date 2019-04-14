# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 08:39:31 2018
@author: Jeyan
"""
import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

print(os.getcwd())
os.chdir(r'C:\Users\murug\Desktop')

data = pd.read_csv('spam.csv', encoding='latin-1')

print(data.head())
print(data.dtypes)
print(data.describe())
print(data.info())

data.columns
data.describe()
words = []
c = len(data)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test = train_test_split(data.v2,
                                                                  data.v1,test_size=0.2,random_state=0)

print(data_train.head(), label_train.head())


def GetVocabulary(datad):
    voc_set = set()
    for email in datad:
        words = email.split()
        for word in words:
            voc_set.add(word)
    return list(voc_set)
vocab_list = GetVocabulary(data_train)
print('Total number of unique words: ', str(len(vocab_list)))

vocab_list.count

vectorizer = CountVectorizer()

def Document2Vector(vocab_list, datad):
    word_vector = np.zeros(len(vocab_list))
    words = datad.split()
    for word in words:
        if word in vocab_list:
            word_vector[vocab_list.index(word)] += 1
return word_vector

print (data_train[1:2,])
print (data_train.values[2])


train_matrix = []

for document in data_train.values:
    word_vector = Document2Vector(vocab_list, document)
    train_matrix.append(word_vector)
print (len(train_matrix))
#--------
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
example = ['I love you, good bad bad', 'you are soo good']\

result = vect.fit_transform(example)
print(result)
print (vect.vocabulary_)
print('\n')

result1 = vect.transform(example)
print(result1)
print (vect.vocabulary_)

vectorizer = CountVectorizer()


data_train_count = vectorizer.fit_transform(data_train)
data_test_count = vectorizer.transform(data_test)
print (data_train_count.shape)
print (data_test_count.shape)
print (vectorizer.vocabulary_)

#------------


word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(),
'occurrences':data_train_count.toarray().sum(axis=0)})
    
word_freq_df['frequency'] = word_freq_df['occurrences'] / np.sum(word_freq_df['occurrences'])

plt.plot(word_freq_df.occurrences)

plt.show()

word_freq_df_sort = word_freq_df.sort_values(by=['occurrences'], ascending=False)

print(word_freq_df_sort.head())


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


model = MultinomialNB()
model.fit(data_train_count, label_train)
predictions = model.predict(data_test_count)

print(predictions[1:50])


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

print (accuracy_score(label_test, predictions))
print (classification_report(label_test, predictions))
print (confusion_matrix(label_test, predictions))

cross_val = cross_val_score(model, data_train_count, label_train, cv=20, scoring='accuracy')
print (cross_val)
print (np.mean(cross_val))



