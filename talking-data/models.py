import pickle
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

from features import FeatureVectorizer, CategoryEncoder
from features import DenseTransformer, TimeAppUsedTransformer


class BaseModel(BaseEstimator):
    """
    Base class for exposing models interface.
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

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)

    def dump(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def get_classes(self):
        """Returns the resulting classes used for the estimator"""
        return self.pipeline.named_steps['classifier'].classes_

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            self = pickle.load(f)
        return self


class DummyEstimator(BaseModel):
    """
    Dummy model used as baseline.
    """
    name = 'Dummy Baseline'

    def fit(self, X, y):
        return self

    def predict(self, X):
        prob = 1./12
        return np.array([[prob] * 12] * len(X))

    def score(self, X, y):
        return 1


class PhoneBrandEstimator(BaseModel):
    """
    Simple estimator using phone brand and model
    """
    name = 'LREstimator'

    def __init__(self):
        self.pipeline = Pipeline([
            ('union', make_union(
                make_pipeline(
                    FeatureVectorizer('phone_brand'),
                    TfidfTransformer(),
                ),
                make_pipeline(
                    FeatureVectorizer('device_model'),
                    TfidfTransformer(),
                ),
            )),
            ('classifier', LogisticRegression(solver='lbfgs',
                                              multi_class='multinomial',
                                              warm_start=True,
                                              n_jobs=-1)),
        ])


class XGBEstimator(BaseModel):
    """."""
    name = 'XGBEstimator'

    def __init__(self):
        self.pipeline = Pipeline([
            ('union', make_union(
                make_pipeline(
                    FeatureVectorizer('phone_brand'),
                    TfidfTransformer(),
                ),
                make_pipeline(
                    FeatureVectorizer('device_model'),
                    TfidfTransformer(),
                ),
                CategoryEncoder('app_id'),
                # PCA(),
            )),
            # ('timetransformer', TimeAppUsedTransformer('timestamp')),
            # densifier must go just before the classifier
            ('densifier', DenseTransformer()),
            ('classifier', xgb.XGBClassifier()),
        ])

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
