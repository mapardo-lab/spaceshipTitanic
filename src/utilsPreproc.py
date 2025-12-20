import pandas as pd

def preprocess_features01(df: pd.DataFrame): 
    """
    Performs initial feature engineering and cleaning on the dataset.

    This function creates new features from existing ones and handles specific
    missing data scenarios based on domain knowledge.

    - Splits the 'Cabin' column into 'Deck', 'Num', and 'Side'.
    - Splits the 'Name' column into 'FirstName' and 'Surname'.
    - For passengers in 'CryoSleep', it assumes they did not use any amenities,
      so it sets 'ShoppingMall', 'VRDeck', 'RoomService', and 'FoodCourt' to 0.

    Args:
        df (pd.DataFrame): The input DataFrame with raw features.

    Returns:
        pd.DataFrame: A new DataFrame with the preprocessed features.
    """
    df_out = df.copy()
    df_out[['Deck','Num','Side']] = df_out['Cabin'].str.split('/', expand = True)
    df_out[['FirstName','Surname']] = df_out['Name'].str.split(' ', expand = True)
    df_out.loc[~df_out['CryoSleep'].isna() & df_out['CryoSleep'], ['ShoppingMall', 'VRDeck', 'RoomService', 'FoodCourt']] = 0
    return df_out

def preprocess_target01(df: pd.DataFrame): 
    """
    Prepares the target variable for the model.

    This function creates the 'target' column required for model training and
    evaluation. In this specific preprocessing step, it simply duplicates the
    'Transported' column.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with the 'target' column added.
    """
    df_out = df.copy()
    df_out['target'] = df_out['Transported']
    return df_out
