# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 02:38:49 2022

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

# DATA PROCESSING
Company_data = pd.read_csv("Company_Data.csv")
Company_data

Company_data.shape
Company_data.head

Company_data.info()  # list of the variable names with the data type it is 
Company_data.isnull().sum()     # finding missing values are their are not 

list(Company_data)

# ================== DATA TRANSFORMATION ===================

# Transforming from Categorical data to Numerical data 

from sklearn import preprocessing 
lable_encoder = preprocessing.LabelEncoder()
Company_data["ShelveLoc"]=lable_encoder.fit_transform(Company_data["ShelveLoc"])
Company_data["Urban"]=lable_encoder.fit_transform(Company_data["Urban"])
Company_data["US"]=lable_encoder.fit_transform(Company_data["US"])

Company_data
Company_data.head()
Company_data.info()

# Transforming from Numerical data to Categorical data

Company_data["Sales"].max()
Company_data["Sales"].min()
Company_data["Sales"].mean()

def fun1(text):
    if text > 7.49:
        return"High"
    else:
        return"low"

Company_data["Sales_new"]=Company_data["Sales"].apply(fun1)   
Company_data["Sales_new"]        

Company_data["Sales_new"].value_counts()   # It counts the number high and low 

# <<<<< EXPLORATION DATA ANALYSIS <<<<<<<

# Scatter plot between the variables along with histograms

import seaborn as sns
sns.pairplot(Company_data)

# SPLITTING THE DATE 

X = Company_data.iloc[:,1:11]
X.info()

Y=Company_data["Sales_new"]
Y

# DATA PARTIANTON 

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=11)

#>>>>>>>>>>>>>>>>>> RANDOM CLASSIFIER <<<<<<<<<<<<<<<<<<<

from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(
                      max_features=0.6,
                      n_estimators=500)

RF.fit(X_train,Y_train)
Y_pred=RF.predict(X_test)
metrics.accuracy_score(Y_test, Y_pred).round(2)

# By applying the Randomforest classifier we could see that 81% accuracy with 60%
# of variables only used.

Traning_Accuracy=[]
Test_Accuracy=[]

for i in range(1,11):
    classifier = RandomForestClassifier(max_depth=i) 
    classifier.fit(X_train, Y_train)
    Y_pred_train = classifier.predict(X_train) 
    Y_pred_test = classifier.predict(X_test) 
    Traning_Accuracy.append(metrics.accuracy_score(Y_train, Y_pred_train).round(2))
    Test_Accuracy.append(metrics.accuracy_score(Y_test, Y_pred_test).round(2))

print(Traning_Accuracy)
print(Test_Accuracy)

# Above single decsion tree accuracies will not be stable for all the time
# So i implemented other models like bagging classifier and random forest classifier 
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

# bagging with gini method we could see that same 78% accuracy as consistent
# bagging with entropy method we could see that 78% accuracy as consistent

#>>>>>>>>>>>>>>>>>>>>>>>> CONCLUSION <<<<<<<<<<<<<<<<<<<<<<<
   
# By applying all the above methods we found that minimum complexity (60%) of Random forests,
# we are getting as 81% accuracy score as the best 


