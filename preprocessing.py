import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(train, test):
    train['Churn'] = [0 if item == 'No' else 1 for item in train['Churn']]
    train = train.replace(pd.np.nan, 0)
    train = FunLabelEncoder(train)

    test = test.drop(columns=['Churn'], axis=1)
    test = test.dropna(how='any')

    Y = train['Churn']
    X = train.drop(columns=['Churn'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=9)

    return X_train, X_test, Y_train, Y_test

def FunLabelEncoder(df):
    le = LabelEncoder()
    for c in df.columns:
        if df.dtypes[c] == object:
            le.fit(df[c].astype(str))
            df[c] = le.transform(df[c].astype(str))
    return df