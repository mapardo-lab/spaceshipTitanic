import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import GridSearchCV

def run_experiments(dataproc, experiments, X_train, y_train):
    """
    Run grid search experiments with different feature sets and machine learning models.

    This function performs the following steps for each combination of feature set and model:
    1. Selects features from the training data
    2. Performs grid search with cross-validation
    3. Records performance metrics for all parameter combinations
    """
    results = []
    for dp, features in dataproc.items():
        X_train_sel = X_train[features]    
        for index, model in enumerate(experiments['model']):
            params = experiments['parameters'][index]
            algorithm = experiments['algorithms'][index]
            grid = GridSearchCV(algorithm, param_grid = params, cv = 5, 
                                scoring= 'accuracy', return_train_score = True)
            grid.fit(X_train_sel, y_train)

            lst = list(grid.get_params()['param_grid'].values())
            for params, test_score, train_score, fit_time in zip(itertools.product(*lst),
                                                   grid.cv_results_['mean_test_score'], 
                                                   grid.cv_results_['mean_train_score'],
                                                   grid.cv_results_['mean_fit_time']):
                params_dict = {param: value for param, value in zip(grid.get_params()['param_grid'].keys(), params)}
                result = {'feature_name': dp,'features': features, 'model': model, 'parameters': params_dict,
                          'train_score': train_score, 'test_score': test_score, 'fit_time': fit_time}
                results.append(result)
    return(results)
    
def expand_parameters(df):
    """
    Expands a dictionary column named 'parameters' into separate columns in a DataFrame.

    For each row in the input DataFrame, the function takes the dictionary stored in the
    'parameters' column, expands each key-value pair into separate columns, and combines
    them with the original DataFrame.
    """
    # expand dictionary column
    expanded_df = df['parameters'].apply(pd.Series)
    # combine with original DataFrame
    result = pd.concat([df.drop('parameters', axis=1), expanded_df], axis=1)
    return result