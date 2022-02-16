import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
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
'''For all the balance features the lower values have much higher proportion of churning customers
   For most frequent vintage values, the churning customers are slightly higher, while for higher values
   of vintage, we have mostly non churning customers which is in sync with the age variable '''
# Drop columns that won't be needed.
y_target = customers_data.churn
customers_data = customers_data.drop(['churn', 'customer_id', 'occupation'], axis=1)

baseline_columns = ['current_month_credit', 'previous_month_credit', 'current_balance',
    'previous_month_end_balance', 'vintage', 'occupation_retired', 'occupation_salaried', 'occupation_self_employed',
    'occupation_student']

customers_data_bs = customers_data[baseline_columns]

# split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(customers_data_bs, y_target,\
     test_size=0.3, random_state=42, stratify=y_target) 

# instantiate and fit the model 
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Plottng roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label="Validation AUC-ROC="+str(auc))
x = np.linspace(0, 1, 1000)
plt.plot(x, x, linestyle='-')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
# plt.show()
'''From the AUC-ROC curve interpreting the model, if I take about top 20% of the population I'll
get more than 60% of the customer that will churn.'''

# Confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    norm_cm =  cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=['Predicted: No', 'Predicted: Yes'], \
            yticklabels=['Actual: No', 'Actual: Yes'], cmap=cmap)


plot_confusion_matrix(cm, ['No', 'Yes'])
plt.show()

recall_score(y_test, y_pred)

# cross validation using StratifiedKFold
def cv_score(ml_model, rstate = 42, thres = 0.5, cols = customers_data.columns):
    i = 1
    cv_scores = []
    cus_data = customers_data.copy()
    cus_data = customers_data[cols]

    # 5 fold cross validation stratified on the basis of target.
    skf = StratifiedKFold(n_splits=5, random_state=rstate, shuffle=True)
    for cus_index, test_index in skf.split(cus_data, y_target):
        print('\n{} of kfold {}'.format(i,skf.n_splits))
        xtrain, xval = cus_data.loc[cus_index], cus_data.loc[test_index]
        ytrain, yval = y_target.loc[cus_index], y_target.loc[test_index] 

        # instantiate model for fitting the data.
        model = ml_model
        model.fit(xtrain, ytrain)
        ypred_prob = model.predict_proba(xval)
        p_values = []

        # using threshold to define the class based on probability
        for j in ypred_prob[:, 1]:
            if j > thres:
                p_values.append(1)
            else:
                p_values.append(0)

        # calculate scores for each fold and print
        pred_values = p_values 
            
                
