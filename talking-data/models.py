import pickle
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, make_union
from sklearn.ensemble import RandomForestClassifier
from features import FeatureVectorizer


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
        return self.pipeline.named_steps['randomforestclassifier'].classes_

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
        return np.array([self.mode] * len(dataset))

    def score(self, X, y):
        return 1


class PhoneBrandEstimator(BaseModel):
    """
    Simple estimator using phone brand and model
    """
    name = 'Phone brand and model estimator'

    def __init__(self):
        self.pipeline = make_pipeline(
            make_union(
                make_pipeline(
                    FeatureVectorizer('phone_brand',
                    ),
                ),
                make_pipeline(
                    FeatureVectorizer('device_model',
                    ),
                ),
            ),
            RandomForestClassifier(n_estimators=10,
                                   n_jobs=-1,
                                   random_state=42
            )
        )
