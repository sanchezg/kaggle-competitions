import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class CategoryEncoder(TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, X, y):
        return self

    def transform(self, X):
        return pd.get_dummies(X[self.column])

    def fit_transform(self, X, y=None):
        return X


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


class TimeAppUsedTransformer(TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None, *args):
        return self

    def transform(self, X, *args):
        def time_of_day(hour):
            tod = ''
            if 6 < hour.hour <= 11:
                tod = 'morning'
            elif 11 < hour.hour <= 13:
                tod = 'midday'
            elif 13 < hour.hour <= 18:
                tod = 'afternoon'
            elif 18 < hour.hour <= 23:
                tod = 'evening'
            elif 23 < hour.hour <= 1:
                tod = 'midnight'
            elif 1 < hour.hour <= 6:
                tod = 'night'
            return tod
        X['tod'] = X[self.column].apply(lambda x: time_of_day(x))
        return X


class DenseTransformer(TransformerMixin):
    """
    Densifier used only in the prediction stage.
    """
    def fit(self, X, y=None, *args):
        return self

    def transform(self, X, *args):
        return X.toarray()

    def fit_transform(self, X, y=None, *args):
        return X
