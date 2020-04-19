from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string

def process_text(text):
    digits=['0','1','2','3','4','5','6','7','8','9']
    nopunc = [char for char in text if char not in string.punctuation and char not in digits ]
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

classifier = RandomForestClassifier(n_estimators = 300)
classifier.fit(X_train, Y_train)

#from sklearn.metrics import accuracy_score
#pred = classifier.predict(X_train)
#print('Accuracy: ', accuracy_score(Y_train,pred))

#pred2= classifier.predict(X_test)
#print('Accuracy for test:', accuracy_score(Y_test, pred2))

