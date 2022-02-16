import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
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


# Building a baseline model.
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


plot_confusion_matrix(cm)

recall_score(y_test, y_pred)
'''From the confusion matrix you can see that we are more focused on those that actually churn and was
predicted as True positive and also those that churn but were predicted as false negative.
Therefore getting a very low recall score doesn't make the model good'''

# Cross validation using StratifiedKFold
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
        roc_score = roc_auc_score(yval, ypred_prob[:, 1])
        recall = recall_score(yval, pred_values)
        precision = precision_score(yval, pred_values)
        accuracy = accuracy_score(yval, pred_values)
        sufix = ""
        msg = ""
        msg += "ROC AUC SCORE: {}, Recall Score: {:.4f}, Precision Score: {:.4f}, Accuracy Score: {:.4f}".format(roc_score, \
            recall, precision, accuracy)
        print("{}".format(msg))

        cv_scores.append(roc_score)
        i += 1
    
    return cv_scores

baseline_score = cv_score(LogisticRegression(), cols = baseline_columns)
all_features_score = cv_score(LogisticRegression())
'''It can clearly be seen that the metric scores increased as the variables increased. This tell 
that there is still more information in the data which were not used in the baseline model.'''
            
                
# Using Recursive Feature Elimination or Backward Selection.
model = LogisticRegression()
rfe = RFE(estimator=model, n_features_to_select=1, step=1)
rfe.fit(customers_data, y_target)

ranking_cus_data = pd.DataFrame()
ranking_cus_data['Feature_name'] = customers_data.columns
ranking_cus_data['Rank'] = rfe.ranking_

ranked = ranking_cus_data.sort_values(by=['Rank'])

# Build the model using the first 15 ranked values.
# rfe_top_15_scores = cv_score(LogisticRegression(), cols = ranked['Feature_name'][:15].values)
'''From this my metric score increased (Recall precisely) but it is still quite low. Therefore
working on the threshold to get a better recall since AUC-ROC depends on the predicted probabilities
and it is not affected by the threshold.'''

# Better model
rfe_score_15 = cv_score(LogisticRegression(), cols = ranked['Feature_name'][:15].values, thres=0.14)

# Comparing models for better clarification
results_df = pd.DataFrame({"baseline": baseline_score, 'all_features': all_features_score, \
    'rfe_score_15': rfe_score_15})

results_df.plot(y=['baseline', 'all_features', 'rfe_score_15'], kind='bar')
plt.show() 
'''From the plot it is clearly seen that the rfe model performs better than the other model.'''