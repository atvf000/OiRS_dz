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

    cv = skl_text.CountVectorizer(lowercase=True, ngram_range=(1,1), strip_accents='unicode', stop_words={'english'})
    cv.fit(data['comment_text'])

    X = cv.transform(teach_data["comment_text"])
    X_predict = cv.transform(test_data["comment_text"])
    y = teach_data['toxic']

    return X, y, X_predict


def start():
    data = pd.read_csv("train.csv")
    X, y, X_predict = prepare(data)
    
    fitted_regression = skl_lm.LogisticRegression(max_iter=1000).fit(X, y)
    y_predict = fitted_regression.predict(X)
    
    cr = skl_m.classification_report(y, y_predict)
    f1 = skl_m.f1_score(y, y_predict, average='macro')
    
    print("Classification_report\n ", cr)
    print("F1:", f1)

    f1_scorer = skl_m.make_scorer(skl_m.f1_score, average='macro')
    grid_search = skl_ms.GridSearchCV(skl_lm.LogisticRegression(max_iter=1000),
                               dict(C=np.arange(0.01, 1, 0.1)),
                            scoring=f1_scorer)

    grid_search.fit(X, y)
    y_predict_grid_search = grid_search.predict(X_predict)
    cr_grid_search = skl_m.classification_report(y, y_predict_grid_search)
    f1_grid_search = skl_m.f1_score(y, y_predict_grid_search, average='macro')


    print("Classification_report grid search: ", cr_grid_search)
    print("F1 grid search:", f1_grid_search)


if __name__ == '__main__':
    start()
