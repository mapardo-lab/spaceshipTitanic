import pandas as pd
import numpy as np
import time
import itertools
from itertools import product
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

def run_experiments(dataproc, experiments, X_train, y_train, scoring = 'accuracy', n_jobs = 1):
    """
    Run grid search experiments with different feature sets and machine learning models.

    This function performs the following steps for each combination of feature set and model:
    1. Selects features from the training data
    2. Performs grid search with cross-validation
    3. Records performance metrics for all parameter combinations
    """
    results = []
    for dp, features in tqdm(dataproc.items(), desc="Datasets analysis"):
        X_train_sel = X_train[features]    
        for index, model in enumerate(experiments['model']):
            params = experiments['parameters'][index]
            algorithm = experiments['algorithms'][index]
            grid = GridSearchCV(algorithm, param_grid = params, cv = 5, 
                                scoring= scoring, return_train_score = True, n_jobs = n_jobs)
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
    
def time_gridsearch(grid, features, X_train, y_train):
    """
    Execute a GridSearchCV operation and measure its execution time.

    This function performs a grid search cross-validation fit operation
    on the specified features and returns both the execution time
    and the number of parameter combinations tested.
    """
    start_time = time.time()
    X_train_sel = X_train[features]
    grid.fit(X_train_sel, y_train)
    end_time = time.time()
    execution_time = end_time - start_time
    num_tests = len(grid.cv_results_['params'])
    return execution_time, num_tests
    
def constrained_combinations(list_of_lists):
    """
    Generate constrained combinations from a list of lists.

    This function computes all possible combinations of elements from input lists
    with the following constraints:
    1. At least one element must be selected from at least two different input lists
    2. None values are excluded from the final combinations
    3. Results are flattened into single-level lists
    """
    # Generate all possible combinations (including None for "no element")
    expanded = [lst + [None] for lst in list_of_lists]
    
    # Compute Cartesian product (all possible combinations)
    all_combinations = product(*expanded)
    
    # Filter combinations to:
    # 1. Exclude None-only selections
    # 2. Remove None placeholders
    # 3. Ensure at least 2 elements
    # 4. Flatten combinations
    result = [
        [item for sublist in combo
         for item in (sublist if isinstance(sublist, list) else [sublist]) if item is not None]
        for combo in all_combinations
        if sum(1 for item in combo if item is not None) >= 2
    ]
    
    return result