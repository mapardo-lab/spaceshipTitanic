import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class SumFeatureCreator(BaseEstimator, TransformerMixin):
    """
    A custom transformer that creates a new feature by summing specified existing features.

    This transformer is useful for feature engineering, for example, to aggregate
    related numerical features into a single summary feature. It is compatible
    with scikit-learn pipelines.

    Attributes:
        features_to_sum (list of str): The names of the features to be summed.
        new_feature_name (str): The name for the newly created feature.
    """
    def __init__(self, features_to_sum, new_feature_name):
        self.features_to_sum = features_to_sum
        self.new_feature_name = new_feature_name
        
    def fit(self, X, y=None):
        """
        Fit method (does nothing, just returns self).

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series, optional): Target variable. Ignored.

        Returns:
            self: The instance itself.
        """
        print(X)
        return self
    
    def transform(self, X):
        """
        Adds a new column to the DataFrame which is the sum of specified columns.

        Args:
            X (pd.DataFrame): The data to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with the new summed feature.
        """
        X = X.copy()
        # Create sum feature
        print(X)
        X[self.new_feature_name] = X[self.features_to_sum].sum(axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns the feature names after transformation.

        Args:
            input_features (list of str, optional): Names of the input features.

        Returns:
            list of str: The list of original feature names plus the new feature name.
        """
        if input_features is None:
            input_features = self.feature_names_in_ or []
        return list(input_features) + [self.new_feature_name]
    
class ExcludeLevels(BaseEstimator, TransformerMixin):
    """
    A transformer that converts a categorical feature into a binary representation.

    It checks if each value in the input feature is present in a predefined list
    of excluded levels. It outputs 1 if the value is *not* in the excluded list,
    and 0 if it is. This is useful for creating new binary features that
    highlight the presence or absence of specific categories.

    Attributes:
        excluded (list): A list of categorical levels to be mapped to 0.
    """
    def __init__(self, excluded):
        self.excluded = excluded

    def fit(self, X, y=None):
        """
        Fit method (does nothing, just returns self).

        Args:
            X (pd.DataFrame or np.ndarray): Input features.
            y (pd.Series, optional): Target variable. Ignored.

        Returns:
            self: The instance itself.
        """
        return self

    def transform(self, X):
        """
        Transforms the input feature into a binary representation.

        Args:
            X (pd.Series or np.ndarray): The input data to transform.

        Returns:
            np.ndarray: A NumPy array of integers (0 or 1).
        """
        return (~np.isin(X, self.excluded)).astype(int)

    def get_feature_names_out(self, input_features=None):
        """
        Returns the feature names after transformation.
        """
        if input_features is None:
            input_features = self.feature_names_in_ or []
        return list(input_features)

class TransformMetadata:
    """
    A factory class for creating and retrieving pre-defined scikit-learn
    preprocessing pipelines.

    This class centralizes the construction of complex `ColumnTransformer` objects,
    allowing different preprocessing strategies to be versioned and accessed by a
    simple name. It is designed to be used statically without instantiation.
    """
    
    @staticmethod
    def get_transformation(name):
        """
        Retrieves a pre-configured scikit-learn ColumnTransformer by name.

        This static method acts as a factory for complex preprocessing pipelines.
        Each named transformation is a `ColumnTransformer` that applies a series
        of specific steps to different subsets of columns in a DataFrame.

        Args:
            name (str): The identifier for the desired transformation pipeline.

        Returns:
            ColumnTransformer: An instance of a scikit-learn `ColumnTransformer`.

        Raises:
            ValueError: If the specified `name` does not correspond to an available transformation.
        """
        expenses = ['ShoppingMall', 'VRDeck', 'RoomService', 'FoodCourt']
        num = ['Age'] + expenses

        pipeline1 = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('create_sum', SumFeatureCreator(features_to_sum = expenses, new_feature_name = 'TotalExpenses')),
            ('scaler', StandardScaler())
        ])

        cat = ['CryoSleep','HomePlanet', 'Destination', 'VIP', 'Deck', 'Side']
        pipeline2 = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehotencoder', OneHotEncoder(sparse_output=False, drop='if_binary'))
        ])

        pipeline3 = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('exclude_cef', ExcludeLevels(excluded = ['C', 'E', 'F'])),
        ])

        pipeline4 = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('exclude_bcef', ExcludeLevels(excluded = ['B','C', 'E', 'F'])),
        ])

        transformations = {
            'transform01': ColumnTransformer(
                transformers = [
                    ('scaled', pipeline1, num),
                    ('OHE', pipeline2, cat),
                    ('exclude_cef', pipeline3, ['Deck']),
                    ('exclude_bcef', pipeline4, ['Deck'])
                ]
            )
        }

        if name not in transformations:
            raise ValueError(f"Unknown transformation: {name}. "
                           f"Available: {list(transformations.keys())}")
        
        return transformations[name]
