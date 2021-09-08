# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 22:00:22 2021

@author: sahil
"""

# Import Libraries
import numpy as np
import pandas as pd
import imblearn as imb_learn
import sklearn as sk


whole_data = pd.read_csv('ks-projects-201801.csv', parse_dates=['launched','deadline'])

# The missing value processing
whole_data = whole_data.fillna({'name':'unknown'})

# Drop the index where state is 'undefined', 'suspended', 'canceled' 
whole_data['state'].replace('undefined','suspended', inplace=True)
whole_data['state'].replace('canceled','suspended', inplace=True)
whole_data['state'].replace('suspended','NaN', inplace=True)
data = whole_data[whole_data['state'] != 'NaN']

# Create launched year column
launched_year = data.launched.dt.year
data = data.assign(launched_year = data.launched.dt.year)

# Drop columns that are unnecessary
unnecessary_columns = ['name','usd pledged', 'currency', 'category', 'deadline', 'launched', 'goal', 'pledged'] 
data.drop(unnecessary_columns, axis=1,inplace=True)

# One Hot Encoding
category_binary = pd.get_dummies(data[['ID','main_category']],prefix='cat')
country_binary = pd.get_dummies(data[['ID','country']],prefix='country')
final_data = data
final_data['state'] = np.where(final_data['state']=='successful',1,0)
final_data = final_data.drop(['main_category','country','usd_pledged_real'],axis=1)
final_data = final_data.merge(category_binary,on='ID').merge(country_binary,on='ID')
final_data = final_data.drop('ID',axis=1)

state_1 = final_data[final_data['state'] == 1].value_counts().sum()/final_data['state'].value_counts().sum()
state_0 = final_data[final_data['state'] == 0].value_counts().sum()/final_data['state'].value_counts().sum()
y_imbalance = [state_1,state_0]
state_mean = final_data.groupby('state').mean()

X = final_data.loc[:, final_data.columns != 'state']
y = final_data.loc[:, final_data.columns == 'state']

# Train-Test split
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.3,random_state=99)

# Fitting Logistic Regression Classifier
print('Logistic Regression Classifier - ')
logregression = sk.linear_model.LogisticRegression().fit(X_train,y_train)
y_hat_logreg = logregression.predict(X_test)
confusion_matrix_logreg = sk.metrics.confusion_matrix(y_test, y_hat_logreg)
print('Confusion Matrix:')
print(confusion_matrix_logreg)
success_logreg = (confusion_matrix_logreg[0,0]+confusion_matrix_logreg[1,1])/len(y_test)
error_logreg = (confusion_matrix_logreg[0,1]+confusion_matrix_logreg[1,0])/len(y_test)
coeff_logreg = logregression.coef_[0]
print('Variable Importance:')
for i,v in enumerate(coeff_logreg):
	print('Feature: %s, Score: %.5f' % (X_train.columns[i],v))
print('---'*30)
     
# Fitting Decision Tree Classifier
print('Decision Tree Classifier - ')
tree = sk.tree.DecisionTreeClassifier().fit(X_train, y_train)
y_hat_tree = tree.predict(X_test)
confusion_matrix_tree = sk.metrics.confusion_matrix(y_test, y_hat_tree)
print('Confusion Matrix:')
print(confusion_matrix_tree)
success_tree = (confusion_matrix_tree[0,0]+confusion_matrix_tree[1,1])/len(y_test)
error_tree = (confusion_matrix_tree[0,1]+confusion_matrix_tree[1,0])/len(y_test)
coeff_tree = tree.feature_importances_
print('Variable Importance:')
for i,v in enumerate(coeff_tree):
	print('Feature: %s, Score: %.5f' % (X_train.columns[i],v))
print('---'*30)

# Fitting Random Forest Classifier
print('Random Forest Classifier - ')
rf = sk.ensemble.RandomForestClassifier(max_features=10).fit(X_train, y_train)
y_hat_rf = rf.predict(X_test)
confusion_matrix_rf = sk.metrics.confusion_matrix(y_test, y_hat_rf)
print('Confusion Matrix:')
print(confusion_matrix_rf)
success_rf = (confusion_matrix_rf[0,0]+confusion_matrix_rf[1,1])/len(y_test)
error_rf = (confusion_matrix_rf[0,1]+confusion_matrix_rf[1,0])/len(y_test)
coeff_rf = rf.feature_importances_
print('Variable Importance:')
for i,v in enumerate(coeff_rf):
	print('Feature: %s, Score: %.5f' % (X_train.columns[i],v))
print('---'*30)

