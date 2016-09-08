import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class CategoryEncoder(TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, X, y):
        return self

    def transform(self, X):
        return pd.get_dummies(X[self.column])


class FeatureVectorizer(TransformerMixin):

    def __init__(self, column, *args, **kwargs):
        self.column = column
        self.cv = CountVectorizer(*args, **kwargs)

    def fit(self, X, y=None, *args):
        self.cv.fit(X[self.column], y)
        return self

    def transform(self, X):
        return self.cv.transform(X[self.column])


class ColumnExtractor(TransformerMixin):

    def __init__(self, column, reshape=False, to_array=False):
        self.column = column
        self.reshape = reshape
        self.to_array = to_array

    def fit(self, X, y=None, *args):
        return self

    def transform(self, X):
        out = X[self.column].values
        if self.reshape:
            out = out.reshape((-1, 1))
        if self.to_array:
            out = out.toarray()
        return out
