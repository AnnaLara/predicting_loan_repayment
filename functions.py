import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

np.random.seed(2019)


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

def keep_states_drop(df, states_to_keep):
    '''Drop rows where state is not in the states_to_keep list'''
    df['state'] = df['state'].apply(lambda x: 'drop' if (x not in states_to_keep) else x)
    df = df.drop(df[(df['state'] == 'drop')].index)
    
    #reset index
    df = df.reset_index(drop=True)
        
    return df

def preprocess(df):
    '''Preprocess according to EDA's conclusions'''
    df = df.drop(['loan_amount', 'loan_borrowed_inc', 'loan_outstanding_inc', 'Unnamed: 0', 'loan_id',  'length_of_transaction_history'], axis=1)
    df = keep_states_drop(df, ['\"WA\"', '\"CA\"', '\"UT\"', '\"ID\"'])
    return df
    

def logireg(X_train, X_test, y_train, y_test):
    '''Perform Logistic Regression and print results'''
    lr = LogisticRegression(solver='lbfgs', fit_intercept = False, n_jobs=-1,  C = 1e12, random_state=113)
    lr.fit(X_train, y_train)
    report = get_print_report(lr, X_test, y_test)
    return lr, report
    
    
def xgb(X_train, X_test, y_train, y_test):
    '''Perform Gradient Boost and print results'''
    xgb = XGBClassifier(learning_rate=0.1, max_depth=3, min_child_weight=2, n_estimators=125, random_state=113)
    xgb.fit(X_train, y_train)
    report = get_print_report(xgb, X_test, y_test)
    return xgb, report
    
def gb(X_train, X_test, y_train, y_test):
    '''Perform Gradient Boosting and print results'''
    gb = GradientBoostingClassifier(learning_rate=.03, max_depth=4, n_estimators=150, random_state=113)
    gb.fit(X_train, y_train)
    report = get_print_report(gb, X_test, y_test)
    return gb, report
    
def rf(X_train, X_test, y_train, y_test):
    '''Perform Random Forest and print results'''
    rfc = RandomForestClassifier(n_estimators=100, random_state=113)
    rfc.fit(X_train, y_train)
    print(rfc)
    report = get_print_report(rfc, X_test, y_test)
    return rfc, report

def get_print_report(model, X_test, y_test):
    print("Report:")
    y_true, y_pred = y_test, model.predict(X_test)
    CV_score = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')
    report = classification_report(y_true, y_pred) + ' CV score: ' +  str(CV_score.mean())
    print(report)
    confusion_matrix(y_true, y_pred)
    print('\nEach Cross Validated Accuracy: \n', CV_score)
    print("\nOverall Classifier Accuracy: %0.2f (+/- %0.2f)\n" % (CV_score.mean(), CV_score.std() * 2))
    return report
    
def predict_probability_and_class(model, X_test, y_test, theshold_for_class_0):
    '''Returns dataframe with predicted probabilities of classes, predicted class;
        print confusion matrix'''
    y_proba = model.predict_proba(X_test)
    res = pd.DataFrame(y_proba, y_test)
    res = res.reset_index()
    res['pred_status'] = res[0].apply(lambda x: 0 if x > theshold_for_class_0 else 1 )
    y_true, y_pred = res['status'], res['pred_status']
    print(classification_report(y_true, y_pred))
    return res

def check_calibration(res):
    '''Return list of sets with average class, range of predicted_probabilities and count'''
    res_c = res.copy()
    averages = [res_c.loc[res_c[0] < 0.1]['status'].mean(), 
           res_c.loc[(res_c[0] > 0.1) & (res_c[0] < 0.2)]['status'].mean(),
           res_c.loc[(res_c[0] > 0.2) & (res_c[0] < 0.3)]['status'].mean(),
           res_c.loc[(res_c[0] > 0.3) & (res_c[0] < 0.4)]['status'].mean(),
           res_c.loc[(res_c[0] > 0.4) & (res_c[0] < 0.5)]['status'].mean(),
           res_c.loc[(res_c[0] > 0.5) & (res_c[0] < 0.6)]['status'].mean(),
           res_c.loc[(res_c[0] > 0.6) & (res_c[0] < 0.7)]['status'].mean(),
           res_c.loc[(res_c[0] > 0.7) & (res_c[0] < 0.8)]['status'].mean(),
           res_c.loc[(res_c[0] > 0.8) & (res_c[0] < 0.9)]['status'].mean(),
           res_c.loc[(res_c[0] > 0.9) & (res_c[0] < 1)]['status'].mean()]
    lens = [len(res_c.loc[res_c[0] < 0.1]['status']), 
           len(res_c.loc[(res_c[0] > 0.1) & (res_c[0] < 0.2)]['status']),
           len(res_c.loc[(res_c[0] > 0.2) & (res_c[0] < 0.3)]['status']),
           len(res_c.loc[(res_c[0] > 0.3) & (res_c[0] < 0.4)]['status']),
           len(res_c.loc[(res_c[0] > 0.4) & (res_c[0] < 0.5)]['status']),
           len(res_c.loc[(res_c[0] > 0.5) & (res_c[0] < 0.6)]['status']),
           len(res_c.loc[(res_c[0] > 0.6) & (res_c[0] < 0.7)]['status']),
           len(res_c.loc[(res_c[0] > 0.7) & (res_c[0] < 0.8)]['status']),
           len(res_c.loc[(res_c[0] > 0.8) & (res_c[0] < 0.9)]['status']),
           len(res_c.loc[(res_c[0] > 0.9) & (res_c[0] < 1)]['status'])]
    bins = ['0 - 0.1', '0.1 - 0.2', '0.2 - 0.3', '0.3 -0.4', '0.4 - 0.5', '0.5 - 0,6',
           '0.6 - 0.7', '0.7 - 0.8', '0.8 - 0.9', '0.9 - 1']
    return list(zip(averages, bins, lens))
    

  
