# Importing all the neccesarry libraries
import pandas as pd
import re
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


class detect:
    def __init__(self): # ML model Constructor
        # Ignoring the warnings
        warnings.simplefilter("ignore")
        # Loading the data set
        data = pd.read_csv("Language Detection.csv")
        # Value counts
        data["language"].value_counts()
        # Separating the dependent and independent features
        X = data["Text"]
        y = data["Language"]

        self.le = LabelEncoder
        y = self.le.fit_transform(y)

        data_list = []

        for text in X:
            text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
            text = re.sub(r'[[]]', ' ', text)
            text = text.lower()
            data_list.append(text)
        # Creating vectorised words
        self.cv = CountVectorizer()
        X = self.cv.fit_transform(data_list).toarray()
        # Train-test spliting
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
        # Model creation and prediction
        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

    def pred(self, text):
        x = self.cv.transform([text]).toarray()
        lang = self.model.predict(x)
        lang = self.le.inverse_transfrom(lang)

        return lang

    def result(self, input1):
        return self.pred(input1)
