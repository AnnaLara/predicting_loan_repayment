import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

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

def logireg(X_train, X_test, y_train, y_test):
    '''Perform Logistic Regression and print results'''
    lr = LogisticRegression(solver='lbfgs', fit_intercept = False, n_jobs=-1,  C = 1e12)
    lr.fit(X_train, y_train)
    print("Report:")
    y_true, y_pred = y_test, lr.predict(X_test)
    print(classification_report(y_true, y_pred))
    confusion_matrix(y_true, y_pred)
    lr_score = cross_val_score(lr, X_test, y_test, cv=5)
    print('\nEach Cross Validated Accuracy: \n', lr_score)
    print("\nOverall Logistic Regression Classifier Accuracy: %0.2f (+/- %0.2f)\n" % (lr_score.mean(), lr_score.std() * 2))
    
    
def xgb(X_train, X_test, y_train, y_test):
    '''Perform Gradient Boost and print results'''
    xgb = XGBClassifier(learning_rate=0.1, max_depth=3, min_child_weight=2, n_estimators=125)
    xgb.fit(X_train, y_train)
    print("Report:")
    y_true, y_pred = y_test, xgb.predict(X_test)
    print(classification_report(y_true, y_pred))
    confusion_matrix(y_true, y_pred)
    xgb_score = cross_val_score(xgb, X_test, y_test, cv=5)
    print('\nEach Cross Validated Accuracy: \n', xgb_score)
    print("\nOverall XGBoost Classifier Accuracy: %0.2f (+/- %0.2f)\n" % (xgb_score.mean(), xgb_score.std() * 2))
    
def gb(X_train, X_test, y_train, y_test):
    '''Perform Gradient Boosting and print results'''
    gb = GradientBoostingClassifier(learning_rate=.03, max_depth=4, n_estimators=150)
    gb.fit(X_train, y_train)
    print("Report:")
    y_true, y_pred = y_test, gb.predict(X_test)
    print(classification_report(y_true, y_pred))
    confusion_matrix(y_true, y_pred)
    gb_score = cross_val_score(gb, X_test, y_test, cv=5)
    print('\nEach Cross Validated Accuracy: \n', gb_score)
    print("\nOverall Gradient Boosting Classifier Accuracy: %0.2f (+/- %0.2f)\n" % (gb_score.mean(), gb_score.std() * 2))
