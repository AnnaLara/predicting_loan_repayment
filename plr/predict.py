import pickle
import numpy as np
import pandas as pd


# Example input: (0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 20, 1000, 5, 0, 1, 0, 1, 500, 100, 300,
# 3, 100, 0, 300, 2000, 4, 1, 1, 500, 0, 0, 0)

def predict_p(user_input, model, scaler):
    '''Returns probability of class 0 and class 1'''

    # with open('../lr_model.sav', 'rb') as f:
    #model = pickle.load(f)

    # with open('../scaler.pkl', 'rb') as f:
    #scaler = pickle.load(f)

    user_input = np.array(user_input)
    df = pd.DataFrame(user_input).T
    df.iloc[:, 0:30] = df.iloc[:, 0:30].astype(float)
    scaled = scaler.transform(df.iloc[:, 0:30])

    scaled_df = pd.DataFrame(scaled)

    scaled_df = pd.concat([scaled_df, df.iloc[:, 30:]], axis=1, join='inner')
    proba = model.predict_proba(scaled_df)

    return proba

# tested with this input:
# predict((0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 20, 1000, 5, 0, 1, 0, 1, 500, 100, 300,
#          3, 100, 0, 300, 2000, 4, 1, 1, 500, 0, 0, 0))
