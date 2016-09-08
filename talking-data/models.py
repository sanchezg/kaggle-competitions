import pickle
import numpy as np
#import xgboost as xgb

#from collections import defaultdict
#from slugify import slugify
from sklearn.base import BaseEstimator
#from sklearn.linear_model import SGDClassifier
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline, make_union
#from features import TitleVectorizer, ColumnExtractor, CategoryEncoder

class BaseModel(BaseEstimator):
    """
    Base clase for exposing models interface.
    """

    name = ''

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)

    def dump(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            self = pickle.load(f)
        return self


class DummyEstimator(BaseModel):
    """
    This is a dummy model that returns the first mode.
    It's used as a baseline to gather some metrics.
    """
    name = 'Dummy Baseline'
    # target_column = 'top_level_category'
    # mode = None

    def fit(self, X, y):
        # self.mode = X[self.target_column].mode()
        return self

    def predict(self, X):
        return np.array([self.mode] * len(dataset))

    def score(self, X, y):
        return 1


class TitleEstimator(BaseModel):
    """
    Simple estimator that vectorizes the titles and uses a linear_model to
    model the estimator.
    """
    name = 'Simple estimator using titles'

    def __init__(self):
        self.pipeline = make_pipeline(
            TitleVectorizer('title',
                strip_accents='ascii',
                # stop_words=STOPWORDS,
                # max_df=0.85
            ),
            SGDClassifier(loss='hinge', n_jobs=-1)
        )
