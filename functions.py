import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def ohe(df, column):
    '''Return transformed dataframe with one-hot-encoded columns'''
    #if rows were dropped, the reset is necessary
    df = df.reset_index(drop='True')
    category = df[column].values.reshape(-1, 1)
    encoder = OneHotEncoder(drop='first', categories='auto').fit(category)
    ohe = pd.DataFrame(encoder.transform(category).toarray(),
                   columns=encoder.get_feature_names([column]))
    df = df.drop([column], axis=1)
    df = pd.concat([df, ohe], axis=1, join='inner')
    return df