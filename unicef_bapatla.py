# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:09:46 2019

@author: user
"""

# To find out the student attrition after school through a survey conducted
# & not actual data.This would give us a way towards Skill Development.

# Set working directory
import os
os.getcwd()

# Import Dataset
import pandas as pd
Dataset = pd.read_excel("DATASET_KhajipalemHighSch.xlsx")

# Basic exploration of data

Dataset.shape

summary = Dataset.describe() # Only for continuous variables

Dataset.info()

Dataset.columns.tolist()

Dataset.isnull().sum()

Dataset.nunique()

#Data cleansing

# Treating of columns with null values; since most of these columns are not
# required due to absence of sufficient information.


Dataset2=Dataset.drop(columns=["Facilities received by CWSN (Mandatory for CWSN Children)",
                              "Email Address (of Student/Parent / Guardian)",
                              "Is Child having Insurance",
                              "Child Blood Group","No of Uniform sets",
                              "Complete Set of free Text Books",
                              "Free Transport (Yes/No/NA)",
                              "Free Escort Facility (Yes/No/NA)",
                              "MDM Benificiary (Yes/No/NA)",
                              "Child Attended Special Trainngs"])

# Dropping other health realted columns which are irrelevant to scope of study

Dataset2=Dataset2.drop(columns=["Iron & Folic acid(Yes/No)","Deworming tablets(Yes/No)",
                                "VitaminA_ReceivedYN (Yes/No)"])


# Dropping columns which are about the admission/examination process; incomplete

Dataset2=Dataset2.drop(columns=["Sur Name","Father Name","Mother Name",
                                "Date of Admission (YYYY/MM/DD)",
                                "Admission Number","First Language",
                                "Second Language","Studying in Section",
                                "Status of Previous Year (If Studying in Class 1)",
                                "Appeared in Last Annual Examination (Yes/No)",
                                "Passed in Last Annual Examination (Yes/No)",
                                "% of Marks Obtained"])

Dataset2.isnull().sum() # Substatially reduced

# In the dataset, all the categorical variables are already encoded using 
#OneHotEncoder. The sample code is below.
# Encoding the Independent Variable
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()

# All categorical variables are encoded with NA =0,NO = 1,YES = 2
# FAther's occupation : Other =0, Agriculture = 1, Aquaculture= 2
# Parent litreracy status : below SSC = 1, Above = 2
#smartphone : No=1, Yes = 2
#Distance from school: 0-4km = 1 , 5km & above =2
#Career Goals : Other =0, Teacher =1, Govt. Administration =2, Engineer =3,
# Security Forces = 4
# Favorite period in timetable : ALL =0, Language = 1, Math/science = 2,
#Sports = 3
# Interested in college: No =0 , Yes =1

# Model Building

# Importing neccesary libraries

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


y = Dataset.iloc[:,-1]
Dataset = Dataset.iloc[:,:-1]

# spilting dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(Dataset, y, test_size = 0.25, random_state = 0)


#RandomForestClassifier()
M1 = RandomForestClassifier(random_state = 100)
M1_Model = M1.fit(X_train, y_train)
Test_Pred = M1_Model.predict(X_test)

# Variable importance
Var_Importance_Df = pd.concat([pd.DataFrame(M1_Model.feature_importances_), pd.DataFrame(X_train.columns)], axis = 1)
Var_Importance_Df
Var_Importance_Df.to_csv("Var_Importance_Df.csv", index = False)

#Var_Importance_Df2 = round(Var_Importance_Df[0], 3)

VarImp_avg = Var_Importance_Df[0].mean()

#Confusion Matrix
Confusion_Mat = confusion_matrix(y_test, Test_Pred)

#Accuracy
Accuracy= ((Confusion_Mat[0,0] + Confusion_Mat[1,1])/y_test.shape[0])*100

# Classification Model Report
Report = classification_report(y_test, Test_Pred)
print(Report)

# Random Forest using RandomSearch for the best parameters

from sklearn.model_selection import RandomizedSearchCV

param_grid = {'n_estimators': [25, 50, 75], 'max_features': [5, 7, 9, 11], 'min_samples_split' : [1000, 2000]} # param_grid is a dictionary
RF_RS = RandomizedSearchCV(RandomForestClassifier(random_state=100), param_distributions=param_grid,  scoring='accuracy', cv=3)
# Other scoring parameters are available here: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

RF_RS_Model = RF_RS.fit(X_train,y_train)

RF_Random_Search_Df = pd.DataFrame.from_dict(RF_RS_Model.cv_results_)

parameters= RF_RS_Model.best_params_

#Parameters will be used for the next phase

# Naive Bayes Model using only Important Variables

Dataset3= Dataset.drop(columns= ["Gender","Medium","Comfortable with SMARTPHONE",
                                 "Distance from Home to School",
                                 "Child Belongs to(Rural/Urban/ Tribal) Area*"])

# spilting dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(Dataset3, y, test_size = 0.25, random_state = 0)

#Building Naive Bayes Model
  
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predict
y_pred = classifier.predict(X_test)

#Confusion matrix
Confusion_Mat2 = confusion_matrix(y_test, y_pred)

#Accuracy
Accuracy= ((Confusion_Mat2[0,0] + Confusion_Mat2[1,1])/y_test.shape[0])*100

#Report
Report = classification_report(y_test, y_pred)
print(Report)

