# Fraud and Anomaly Detection using Synthetic Transactional Data
## The goal is to develop a method that minimizes false negatives when evaluating new data points.

### Install

This project requires **Python 3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

### Data

The Paysim dataset consists of over 6 million data points, each with 9 features, generated by the Paysim Retail Simulation Software.
The dataset is available on [Kaggle](https://www.kaggle.com/ntnu-testimon/paysim1/data).

**Features**
1. 'type': CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.

2. 'amount': amount of the transaction in local currency.

3. 'nameOrig': customer who started the transaction

4, 'oldbalanceOrg': initial balance before the transaction

5. 'newbalanceOrig': new balance after the transaction

6. 'nameDest': customer who is the recipient of the transaction

7. 'oldbalanceDest': initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).

8. 'newbalanceDest': new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).


**Target Variable**
9. 'isFraud' - This is the transactions made by the fraudulent agents inside the simulation.
In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.

One more feature, 'Step', indicates the fictional time at which the transaction occurred. This may or may not be useful to us.