# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 01:48:56 2022

@author: MAHESH
"""

# REQUIRESD LIBRAYERS 
import pandas as pd 
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

Fraud_check_data = pd.read_csv("Fraud_check.csv")
pd.set_option("display.max_columns", 20)
Fraud_check_data

Fraud_check_data.shape
Fraud_check_data.head

Fraud_check_data.info()  # list of the variable names with the data type it is 
Fraud_check_data.isnull().sum()     # finding missing values are their are not 

list(Fraud_check_data)

# ================== DATA TRANSFORMATION ===================

# Transforming from Categorical data to Numerical data 

from sklearn import preprocessing 
lable_encoder = preprocessing.LabelEncoder()
Fraud_check_data["Undergrad"]=lable_encoder.fit_transform(Fraud_check_data["Undergrad"])
Fraud_check_data["Marital.Status"]=lable_encoder.fit_transform(Fraud_check_data["Marital.Status"])
Fraud_check_data["Urban"]=lable_encoder.fit_transform(Fraud_check_data["Urban"])

Fraud_check_data
Fraud_check_data.head()
Fraud_check_data.info()

# Transforming from Numerical data to Categorical data

def fun1(text):
    if text < 30000:       # # <= 30000 as "Risky" and others are "Good"
        return"Riskey"
    else:
        return"Good"

Fraud_check_data["Tax_In"]=Fraud_check_data["Taxable.Income"].apply(fun1)   
Fraud_check_data["Tax_In"]        

Fraud_check_data["Tax_In"].value_counts()       # It counts number of Riskey and good 

# <<<<< EXPLORATION DATA ANALYSIS <<<<<
# Scatter plot between the variables along with histograms

import seaborn as sns
sns.pairplot(Fraud_check_data)

# SPLITTING THE DATE 

X = Fraud_check_data[Fraud_check_data.columns[[0,1,3,4,5]]]
X.info()                     
list(X)

Y=Fraud_check_data["Tax_In"]
Y

# DATA PARTIANTON 

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=20)

#>>>>>>>>>>>>>> RANDOM FOREST CLASSIFIER <<<<<<<<<<<<<<<<<<<<<<<<

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(max_features=0.6, 
                        n_estimators=500)

RF.fit(X_train, Y_train)
Y_pred = RF.predict(X_test) 
metrics.accuracy_score(Y_test,Y_pred).round(2)

Y_pred_train=RF.predict(X_train)
Y_pred_test=RF.predict(X_test)

# CONFUSION MATRIX

from sklearn import metrics

#TRAIN & TEST ACCURACY SCORES

print("Traning Accuracy:",metrics.accuracy_score(Y_train,Y_pred_train).round(2))
print("Test Accuracy:",metrics.accuracy_score(Y_test,Y_pred_test).round(2))

# By applying the Randomforest classifier we could see that 77% accuracy with 60%
# of variables only used.

Traning_Accuracy=[]
Test_Accuracy=[]

for i in range(1,23):
    classifier = DecisionTreeClassifier(max_depth=i) 
    classifier.fit(X_train, Y_train)
    Y_pred_train = classifier.predict(X_train) 
    Y_pred_test = classifier.predict(X_test) 
    Traning_Accuracy.append(metrics.accuracy_score(Y_train, Y_pred_train).round(2))
    Test_Accuracy.append(metrics.accuracy_score(Y_test, Y_pred_test).round(2))

print(Traning_Accuracy)
print(Test_Accuracy)

# Above single RANDOM FOREST accuracies will not be stable for all the time
# So i implemented other models like bagging classifier and AdaBoostClassifier
# to see stability of the performances and accuracyes scores that which model is giving
# the best accuracyes too be observe

#>>>>>>>>>>>>>>>>>> BAGGING CLASSIFIER <<<<<<<<<<<<<<<<<<<

from sklearn.ensemble import BaggingClassifier
classifier=DecisionTreeClassifier(criterion="gini",max_depth=6)
# classifier=DecisionTreeClassifier(criterion="entropy",max_depth=6)

bag=BaggingClassifier(base_estimator=classifier,
                      max_features=0.9,
                      max_samples=0.6,n_estimators=500)

bag.fit(X_train,Y_train)
Y_pred=bag.predict(X_test)
metrics.accuracy_score(Y_test, Y_pred).round(2)

# By applying bagging method we could see that 78% accuracy as consistent

#>>>>>>>>>>>>>>>>>> ADABOOST CLASSIFIER <<<<<<<<<<<<<<<<<<<

from sklearn.ensemble import AdaBoostClassifier
classifier=DecisionTreeClassifier(criterion="gini",max_depth=5)
# classifier=DecisionTreeClassifier(criterion="entropy",max_depth=6)
ABC= AdaBoostClassifier(learning_rate=0.001,n_estimators=500)
ABC.fit(X_train,Y_train)
Y_pred=ABC.predict(X_test)

from sklearn import metrics
metrics.accuracy_score(Y_pred,Y_test).round(3)

# By applying the AdaBoostClassifier we got accuracy score performance upto 78%. 

#>>>>>>>>>>>>>>>>>>>>>>>> CONCLUSION <<<<<<<<<<<<<<<<<<<<<<<
   
# By applying all the above methods we found that minimum complexity (60%) of AdaBoostClassifier and 
# BaggingClassifierwe are getting as 78% accuracy score as the best 
# when compared with rtandomforest accuracy socor
