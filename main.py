from sklearn import datasets, model_selection as ms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_in_predictor_and_target_data(df, predictor_variable_list, target_variable_list):
    predictor_data, target = [], []
    for row in df[predictor_variable_list].values:
        predictor_data.append(row)
    for row in df[target_variable_list].values:
        target.append(row[0])
    return predictor_data, target


allowable_transaction_types = ['TRANSFER', 'CASH_OUT']
# only TRANSFER and CASH_OUT are ever fraud, so only take those points
df = pd.read_csv("paysim_kaggle_dataset.csv")
df = df.loc[df['type'].isin(allowable_transaction_types)]
predictor_vars = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
target_vars = ['isFraud']

paysim_predictor_data, paysim_target = load_in_predictor_and_target_data(df, predictor_vars, target_vars)

x_train, x_test, y_train, y_test = ms.train_test_split(paysim_predictor_data, paysim_target, test_size=0.25, random_state=None)
logisticRegr = LogisticRegression()  # instance of the logistic regression model with all params as default
logisticRegr.fit(x_train, y_train)  # train the model with 75% of our data

# make predictions on the remaining 25% of the data
predictions = logisticRegr.predict(x_test)

"""
--Confusion Matrix Formatting--
           [ True Positives ] [ False Positives / Type I ]
[ False Negatives / Type II ] [ True Negatives ]
"""
confusion_matrix = confusion_matrix(y_test, predictions)
print('confusion matrix\n', confusion_matrix)

# check accuracy of the model / performance
score = logisticRegr.score(x_test, y_test)
print('Score: ', score)