import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as skl_ms
import sklearn.linear_model as skl_lm
import sklearn.metrics as skl_m
import sklearn.feature_extraction.text as skl_text

def prepare(data):
    # delete unused columns
    data.pop("severe_toxic")
    data.pop("obscene")
    data.pop("threat")
    data.pop("insult")
    data.pop("identity_hate")

    # transform all words to lowercase
    data['comment_text'] = data['comment_text'].str.lower()
    teach_data, test_data = skl_ms.train_test_split(data, test_size=0.2, random_state=5)

    cv = skl_text.CountVectorizer(strip_accents='unicode', stop_words={'english'}, analyzer='word')
    cv.fit(teach_data['comment_text'])

    X = cv.transform(teach_data["comment_text"])
    y = teach_data['toxic']

    return X, y


def start():
    data = pd.read_csv("train.csv")
    X, y = prepare(data)
    
    fitted_regression = skl_lm.LogisticRegression(max_iter=10000).fit(X, y)
    y_predict = fitted_regression.predict(X)
    
    cr = skl_m.classification_report(y, y_predict)
    f1 = skl_m.f1_score(y, y_predict, average='macro')
    
    print("Classification_report: ", cr)
    print("F1:", f1)

    f1_scorer = skl_m.make_scorer(skl_m.f1_score, average='macro')
    gscv = skl_ms.GridSearchCV(skl_lm.LogisticRegression(max_iter=10000), dict(C=np.arange(0.01, 1, 0.1)),
                            scoring=f1_scorer)

    gscv.fit(X, y)
    y_predict_gscv = gscv.predict(X)
    cr_gscv = skl_m.classification_report(y, y_predict_gscv)
    f1_gscv = skl_m.f1_score(y, y_predict_gscv, average='macro')


    print("Classification_report grid search: ", cr_gscv)
    print("F1 grid search:", f1_gscv)


if __name__ == '__main__':
    start()
