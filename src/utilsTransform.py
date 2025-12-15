import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SumFeatureCreator(BaseEstimator, TransformerMixin):
    """Creates a new feature as the sum of specified features"""
    def __init__(self, features_to_sum, new_feature_name):
        self.features_to_sum = features_to_sum
        self.new_feature_name = new_feature_name
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Create sum feature
        X[self.new_feature_name] = X[self.features_to_sum].sum(axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_ or []
        return list(input_features) + [self.new_feature_name]
    

class ExcludeLevels(BaseEstimator, TransformerMixin):
    """
    A transformer that converts a feature into a binary representation,
    indicating whether each value is *not* present in a predefined list of excluded levels.

    This is useful for creating new features that highlight the presence or absence
    of specific categories within a categorical feature.
    """
    def __init__(self, excluded):
        self.excluded = excluded

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (~np.isin(X, self.excluded)).astype(int)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_ or []
        return list(input_features)
