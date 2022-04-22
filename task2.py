import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import re
import logging as log
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report



from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import (
    CountVectorizer,
)

import warnings
warnings.filterwarnings("ignore")


def build_graph():
    data = pd.read_csv('train.csv')
    sb.pairplot(data)
    plt.show()

def clean_html(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def clean_punc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


def keep_alpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


status = {}


def process():
    log.warning('Loading train.csv data...')
    data = pd.read_csv('train.csv')
    rowSums = data.iloc[:, 2:].sum(axis=1)
    clean_comments_count = (rowSums == 0).sum(axis=0)

    status['total'] = len(data)
    status['clean'] = clean_comments_count
    status['labeled'] = len(data) - clean_comments_count

    log.warning('Simplifying text...')
    data['comment_text'] = data['comment_text'].str.lower()
    data['comment_text'] = data['comment_text'].apply(clean_html)
    data['comment_text'] = data['comment_text'].apply(clean_punc)
    data['comment_text'] = data['comment_text'].apply(keep_alpha)

    log.warning('Splitting data...')
    train, test = train_test_split(data, random_state=42, test_size=0.20, shuffle=True)

    log.warning('Vectorizing strings of comments...')
    train_texts = train['comment_text']
    test_texts = test['comment_text']
    vectorizer = CountVectorizer(ngram_range=(1, 1),  strip_accents='unicode', stop_words={'english'})
    vectorizer.fit(data['comment_text'])

    X_train = vectorizer.transform(train_texts)
    y_train = train['toxic']

    X_test = vectorizer.transform(test_texts)
    y_test = test['toxic']


    log.warning('Training first logic regression...')
    logic = LogisticRegression(random_state=0, )
    logic.fit(X_train, y_train)

    log.warning('Predicting first logic regression...')
    predicted = logic.predict(X_test)

    status['cv_report'] = classification_report(y_test, predicted)
    status['f1_score'] = f1_score(y_test, predicted, average='macro')
    status['c'] = logic.C

    log.warning('Training with GridSearchCV...')
    f1_scorer = make_scorer(f1_score, average='macro')
    logic_cv = GridSearchCV(LogisticRegression(random_state=0, penalty='l2'), dict(C=np.arange(0.01, 1, 0.1)),
                            scoring=f1_scorer)

    logic_cv.fit(X_train, y_train)

    log.warning('Predicting with GridSearchCV...')
    predicted_cv = logic_cv.predict(X_test)

    status['f1_score_gscv'] = f1_score(predicted_cv, y_test, average='macro')
    status['c_gscv'] = logic_cv.best_params_["C"]

    log.warning('Job finished! Collecting results')


def reporting():

    print(f'Total number of comments: {status.get("total")}\n'
          f'Number of clean comments: {status.get("clean")}\n'
          f'Number of comments with labels: {status.get("labeled")}\n')

    print(status.get('cv_report'))
    print(f'F1 score: {status.get("f1_score")}\n'
          f'C is: {status.get("c")}\n')

    print(f'GSCV F1 score: {status.get("f1_score_gscv")}\n'
         f'GSCV best C is: {status.get("c_gscv")}\n')


def start():
    process()
    reporting()


if __name__ == '__main__':
    start()
