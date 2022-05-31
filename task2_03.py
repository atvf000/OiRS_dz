import re
from typing import Dict

import pandas as pd
import numpy as np
import sklearn.cluster as sc
import sklearn.metrics as sm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCORES = dict(
    fbeta=lambda y_true, y_pred: sm.fbeta_score(y_true, y_pred, beta=0.5),
    accuracy=lambda y_true, y_pred: sm.accuracy_score(y_true, y_pred),
    f1=lambda y_true, y_pred: sm.f1_score(y_true, y_pred),
    quality=lambda y_true, y_pred: sm.fbeta_score(y_true, y_pred,
                                                  beta=0.5) * 10000,
)

class Resolver:
    def __init__(self, raw_data: pd.DataFrame):
        self.raw_data = raw_data
        self.model = sc.KMeans(
            n_clusters=2,
            random_state=0
        )
        self.pre_data: pd.DataFrame = None
        self.data: pd.DataFrame = None
        self.result: np.ndarray = None
        self.result_metric: Dict[str, float] = dict()

    def extract_features(self):
        data = self.raw_data.copy()
        data[f'f_perc'] = data.request.apply(lambda x: len(re.findall('%', x)))
        data[f'f_sad'] = data.request.apply(lambda x: len(re.findall('\(', x)))
        data[f'f_permax'] = data.request.apply(lambda x: max([s.count('%') for s in x.split('\n')]))
        data[f'f_apos'] = data.request.apply(lambda x: len(re.findall('\'', x)))
        data[f'f_linux'] = data.request.apply(lambda x: len(re.findall('Linux', x)))
        data[f'f_sleep'] = data.request.apply(lambda x: len(re.findall('sleep', x)))
        data[f'f_select'] = data.request.apply(lambda x: len(re.findall('select', x)))
        data[f'f_abobut'] = data.request.apply(lambda x: len(re.findall('about', x)))
        data[f'f_exclam'] = data.request.apply(lambda x: len(re.findall('!', x)))

        del data['request']
        self.pre_data = data

    def _prepare_data(self):
        data = self.pre_data.copy()
        columns: pd.Index = data.columns
        preparer = make_pipeline(
            StandardScaler(),
        )
        prepared_data = preparer.fit_transform(data)
        df = pd.DataFrame(
            prepared_data,
            columns=columns,
        )
        self.data = df

    def print_results(self, y_true):
        for key in SCORES.keys():
            score = SCORES[key](
                y_true,
                self.result.reshape(-1, 1)
            )
            print(f'{key:>16}: \x1b[34m{score:<9.4f}\x1b[0m')

    def fit(self):
        self._prepare_data()
        self.model.fit(self.data)

    def resolve(self):
        self._prepare_data()
        self.result = self.model.predict(self.data)

    def save_results(self):
        df = pd.DataFrame(
            data=self.result,
            columns=['y_true'],
        )
        df.to_csv('biborka/data/result.csv', index=False)


if __name__ == '__main__':
    raw_data = pd.read_csv('biborka/data/train.csv')
    data = raw_data.copy()
    data.pop('y_true')

    resolver = Resolver(data)
    resolver.extract_features()
    resolver.fit()
    resolver.resolve()
    resolver.print_results(raw_data['y_true'])

    raw_data = pd.read_csv('biborka/data/test.csv')

    resolver.raw_data = raw_data
    resolver.extract_features()

    resolver.resolve()
    resolver.save_results()

