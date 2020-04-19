import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import  MultinomialNB
from nltk.corpus import stopwords
import string

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

df = pd.read_csv("spam.csv")



X = df["EmailText"]
Y = df["Label"]
X.head().apply(process_text)

from sklearn.feature_extraction.text import CountVectorizer
messages = CountVectorizer(analyzer=process_text).fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(messages, Y, test_size = 0.20, random_state = 0)


classifier = MultinomialNB()
classifier.fit(X_train, Y_train)


#from sklearn.metrics import accuracy_score
#pred = classifier.predict(X_train)
#print('Accuracy: ', accuracy_score(Y_train,pred))

#pred2= classifier.predict(X_test)
#print('Accuracy for test:', accuracy_score(Y_test, pred2))

