from sklearn import datasets, model_selection as ms
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("paysim_kaggle_dataset.csv")

# data: contains the values
# target names: the names of the target variables
# .data is a list of lists, each of the same length, containing the predictor variables
# .target is the possible labels, or response variables, in order in one list

paysim_predictor_data = []
paysim_target = []

# load data into predictor and target data lists
for row in df[['isFraud']].values:
    paysim_target.append(row[0])
for row in df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].values:
    paysim_predictor_data.append(row)

x_train, x_test, y_train, y_test = ms.train_test_split(paysim_predictor_data, paysim_target, test_size=0.25, random_state=None)
logisticRegr = LogisticRegression()  # instance of the logistic regression model with all params as default
logisticRegr.fit(x_train, y_train)  # train the model with 75% of our data

# make predictions on the remaining 25% of the data
predictions = logisticRegr.predict(x_test)

# check accuracy of the model / performance
score = logisticRegr.score(x_test, y_test)
print('Score: ', score)