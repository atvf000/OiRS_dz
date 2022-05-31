import re
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, fbeta_score
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def accuracy(y_pred, y):
    return accuracy_score(y, y_pred)


def f1(y_pred, y):
    return f1_score(y, y_pred)


def f_beta(y_pred, y):
    return fbeta_score(y, y_pred, beta=0.5)


def prepare(data):
    data['%'] = data['request'].apply(lambda x: len(re.findall(r'%', x)))
    data['('] = data['request'].apply(lambda x: len(re.findall(r'\(', x)))
    data[')'] = data['request'].apply(lambda x: len(re.findall(r'\)', x)))
    data['!'] = data['request'].apply(lambda x: len(re.findall(r'!', x)))
    data['\''] = data['request'].apply(lambda x: len(re.findall(r'\'', x)))

    data['about'] = data['request'].apply(lambda x: len(re.findall(r'about', x)))
    data['burp'] = data['request'].apply(lambda x: len(re.findall(r'burp', x)))
    data['sleep'] = data['request'].apply(lambda x: len(re.findall(r'sleep', x)))
    data['select'] = data['request'].apply(lambda x: len(re.findall(r'select', x)))
    data['union'] = data['request'].apply(lambda x: len(re.findall(r'union', x)))

    return data


def prepare_fit(data):
    del data['request']

    columns: pd.Index = data.columns
    preparer = make_pipeline(
        StandardScaler(),
    )
    prepared_data = preparer.fit_transform(data)
    df = pd.DataFrame(
        prepared_data,
        columns=columns,
    )

    return df


def create_model():
    return KMeans(
        n_clusters=2,
        random_state=0
    )


def results(y, y_pred):
    print(" f1:\t\t", f1(y_pred, y),
          "\n accuracy:\t", accuracy(y_pred, y),
          "\n quality:\t", f_beta(y_pred, y) * 10000)


def save_results(result):
    df = pd.DataFrame(
        data=result,
        columns=['y_true'],
    )
    df.to_csv('result.csv', index=False)


def start():
    print("Read train data from csv")
    data = pd.read_csv('train2.csv')
    y_true = data['y_true']
    data.pop('y_true')

    print("Prepare train data")
    data = prepare(data)
    data = prepare_fit(data)

    model = create_model()
    model.fit(data)
    result = model.predict(data)
    results(y_true, result)

    # -----------------------------------

    print("Read test data from csv")
    data_test = pd.read_csv('test.csv')

    print("Prepare test data")
    data_test = prepare(data_test)
    data_test = prepare_fit(data_test)
    result_test = model.predict(data_test)

    print("Save results")
    save_results(result_test)


if __name__ == '__main__':
    start()
