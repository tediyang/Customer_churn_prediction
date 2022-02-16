import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score, precision_score
pd.set_option('max_column', None)

# load the data (csv file)
customers_data = pd.read_csv('churn_prediction.csv')
shape = customers_data.shape

# Check for missing values in the dataset.
# columns with missing values.
customers_data.isnull().any()
customers_data.isnull().sum()


# Working on missing values
# Gender column
customers_data['gender'].value_counts()
'''There is a good distribution of males and females, arguably missing values cannot be filled
with any of them. Therefore I'll assign the missing values with the value -1 as a separate 
category after converting the categorical variable.'''
conv_gender = {'Male': 1, 'Female': 0}
customers_data.replace({'gender': conv_gender}, inplace=True)
customers_data['gender'] = customers_data['gender'].fillna(-1)

# Dependents, Occupation and City will be filled with the mode.
customers_data['dependents'].value_counts()
customers_data['dependents'] = customers_data['dependents'].fillna(0)
customers_data['occupation'].value_counts()
customers_data['occupation'] = customers_data['occupation'].fillna('self_employed')
customers_data['city'].value_counts()
customers_data['city'] = customers_data['city'].fillna(1020)

# Days since last transaction
tran_days = customers_data['days_since_last_transaction']
tran_days.max(skipna=True)
'''Assumption will be made on this column as this is number of days since last transaction in 1 year.
I'll substitute missing values with aaa value greater than 1 year, probably 450'''
customers_data['days_since_last_transaction'] = customers_data['days_since_last_transaction'].fillna(450)

# Since I'll be working with a linear model, therefore I'll convert occupation to one-hot encoded
# features.
customers_data = pd.concat([customers_data, pd.get_dummies(customers_data['occupation'],\
     prefix=str('occupation'), prefix_sep='_')], axis=1)

customers_data.describe()
'''Checking the data, I observed that there are lot of outliers in the dataset especially
when it comes to previous and current balance features.'''

# using log transformation to deal with outliers.
outlier_cols = ['customer_nw_category', 'current_balance', 'previous_month_end_balance', 'average_monthly_balance_prevQ2',
    'average_monthly_balance_prevQ', 'current_month_credit', 'previous_month_credit', 'current_month_debit',
    'previous_month_debit', 'current_month_balance', 'previous_month_balance']

for i in outlier_cols:
    customers_data[i] = np.log(customers_data[i] + 17000)

# scale the numerical field.
std = StandardScaler()
scaled_data = std.fit_transform(customers_data[outlier_cols])
scaled_df = pd.DataFrame(scaled_data, columns=outlier_cols)

customers_data_cp = customers_data.copy()
customers_data = customers_data.drop(columns= outlier_cols, axis=1)
customers_data = customers_data.merge(scaled_df, left_index=True, right_index=True, how='left')

# Drop columns that won't be needed.
y_target = customers_data.churn
customers_data = customers_data.drop(['churn', 'customer_id', 'occupation'], axis=1)

print(customers_data.head())