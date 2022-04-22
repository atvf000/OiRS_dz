import numpy as np
import pandas as pd
import sklearn.model_selection as skl_ms
import sklearn.linear_model as skl_lm
import sklearn.metrics as skl_m
import sklearn.feature_extraction.text as skl_text


def prepare(data):
    data.pop("severe_toxic")
    data.pop("obscene")
    data.pop("threat")
    data.pop("insult")
    data.pop("identity_hate")

    teach_data, test_data = skl_ms.train_test_split(data, test_size=0.2, random_state=5)

    vectorizer = skl_text.CountVectorizer(lowercase=True, ngram_range=(1, 1), strip_accents='unicode',
                                          stop_words={'english'})
    vectorizer.fit(data['comment_text'])

    X = vectorizer.transform(teach_data["comment_text"])
    y = teach_data['toxic']

    X_test = vectorizer.transform(test_data["comment_text"])
    y_test = test_data['toxic']

    return X, y, X_test, y_test


def start():
    print("Read data from csv")
    data = pd.read_csv("train.csv")

    print("Prepare data")
    X, y, X_test, y_test = prepare(data)

    print("Start regression")
    regression = skl_lm.LogisticRegression(random_state=0, max_iter=1000)
    regression.fit(X, y)

    print("Start predict")
    y_predict = regression.predict(X_test)

    classification_report = skl_m.classification_report(y_test, y_predict)
    f1 = skl_m.f1_score(y_test, y_predict, average='macro')

    print("Classification_report\n ", classification_report)
    print("f1:", f1)

    print("-"*20, "\nStart grid search")

    f1_scorer = skl_m.make_scorer(skl_m.f1_score, average='macro')

    grid_search = skl_ms.GridSearchCV(skl_lm.LogisticRegression(random_state=0, penalty='l2', max_iter=1000),
                                      dict(C=np.arange(0.01, 1, 0.1)),
                                      scoring=f1_scorer)

    grid_search.fit(X, y)

    print("Start predict\n")
    y_test_predict = grid_search.predict(X_test)

    cr_grid_search = skl_m.classification_report(y_test, y_test_predict)
    f1_grid_search = skl_m.f1_score(y_test, y_test_predict, average='macro')

    print("Classification_report for grid search\n ", cr_grid_search)
    print("f1 for grid search:", f1_grid_search)


if __name__ == '__main__':
    start()
