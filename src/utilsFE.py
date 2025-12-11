import pandas as pd
import numpy as np

def process_features(df, preprocessor, encoder, scaler_num):
    # split Cabin feature (takes the form deck/num/side, where side can be either P for Port or S for Starboard)
    df[['Deck','Num','Side']] = df['Cabin'].str.split('/', expand = True)
    # impute expenses by CryoSleep and Age
    expenses_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df.loc[df['CryoSleep'] == True, expenses_features] = 0.0
    df.loc[df['Age'] < 13, expenses_features] = 0.0
    # simple imputation
    # Age, expenses --> median
    # VIP, Deck, Destination, HomePlanet, CryoSleep, Side --> mode
    # Define columns
    num_features = ['Age'] + expenses_features
    cat_features = ['VIP', 'Destination', 'HomePlanet', 'CryoSleep', 'Deck', 'Side']
    df[num_features + cat_features] = pd.DataFrame(
        preprocessor.transform(df),
        columns=num_features + cat_features
    )
    # feature Age < 12 (True/False)
    df['Age12'] = df['Age'].apply(lambda x: 1.0 if x < 12 else 0.0)
    # calculate TotalExpenses
    df['TotalExpenses'] = df[expenses_features].sum(axis=1)
    # log10 transformation for expense features
    expenses_features_total = expenses_features + ['TotalExpenses']
    expenses_features_log10 = list(map(lambda x: 'log10_' + x, expenses_features_total))
    df[expenses_features_log10] = df[expenses_features_total].apply(lambda x: np.log10(np.float64(x + 1)))
    # log10 expense features classification < 1 (True/False)
    for feature in expenses_features_log10:
        new_feature = f'{feature}1'
        df[new_feature] = df[feature].apply(lambda x: 1.0 if x < 1 else 0.0)
    # numerical standarization
    num_features = ['Age'] + expenses_features + expenses_features_log10
    df[num_features] = scaler_num.transform(df[num_features])
    # categorical encoding
    encoded_data = encoder.transform(df[cat_features])
    features_onehot = encoder.get_feature_names_out(cat_features)
    df_encoded = pd.DataFrame(encoded_data, columns=features_onehot)
    df = pd.concat([df, df_encoded], axis = 1)
    return df