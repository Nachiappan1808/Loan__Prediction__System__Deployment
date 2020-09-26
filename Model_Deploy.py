#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from imblearn.under_sampling import NearMiss
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport


# In[44]:


data=pd.read_csv("train_ctrUa4K.csv")
data


# In[45]:


pd.options.display.float_format = '{:,.0f}'.format


# In[46]:


data['Education'] = data['Education'].str.replace(' ','_')


# In[47]:


data['Dependents']=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
data['Gender']=data.Gender.map({'Male':0,'Female':1})
data['Self_Employed']=data.Self_Employed.map({'Yes':1,'No':0})
data['Education']=data.Education.map({'Graduate':0,'Not_Graduate':1})
data['Property_Area']=data.Property_Area.map({'Rural':0,'Urban':1,'Semiurban':2})
data['Married']=data.Married.map({'No':0,'Yes':1})
#data['Credit_History']=data.Credit_History.map({0:'zero',1:'one'})
#data['Loan_Amount_Term']=data.Loan_Amount_Term.map({12:'one',36:'three',60:'five',84:'seven',120:'ten',180:'fifteen',240:'twenty',300:'twentyfive',360:'thirty',480:'forty'})


# In[48]:


for column in ('Gender','Married','Dependents','Self_Employed','Credit_History','Loan_Amount_Term','Property_Area','Education'):
    data[column].fillna(data[column].mode()[0],inplace=True)
for column in ('LoanAmount','CoapplicantIncome','ApplicantIncome'):
    data[column].fillna(data[column].mean(),inplace=True)


# In[49]:


data['Loan_Status']=data.Loan_Status.map({'Y':1,'N':0})
Y=data['Loan_Status'].values
data.drop(['Loan_Status'],axis=1,inplace=True)
X=data[data.iloc[:,1:13].columns]


# In[50]:


X_tr, X_te, y_train, y_test = train_test_split(X, Y, test_size=0.33, stratify=Y)


# In[51]:


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
classifier = GridSearchCV(LogisticRegression(), param_grid,cv=10,scoring='roc_auc',return_train_score=True)
classifier.fit(X_tr, y_train)


# In[52]:


best_param=classifier.best_params_
print("Best Hyperparameter: ",best_param)
p_C=best_param['C']


# In[53]:


from sklearn.metrics import roc_curve, auc


Log_model = LogisticRegression(C=p_C)
Log_model.fit(X_tr, y_train)



# In[54]:


# Creating a pickle file for the classifier
import pickle
filename = 'loan-prediction-lr-model.pkl'
pickle.dump(Log_model, open(filename, 'wb'))


# In[55]:


#y_test_predict=predict_with_best_t(y_test_pred[:,1], best_t)
y_test_predict=Log_model.predict(X_te)
print("Recall for logistic regression model:",metrics.recall_score(y_test,y_test_predict))
print("Precision for logistic regression model:",metrics.precision_score(y_test,y_test_predict))
print("Accuracy for logistic regression model:",metrics.accuracy_score(y_test,y_test_predict))
print("F-score for logistic regression model:",metrics.f1_score(y_test,y_test_predict))
print("Log-loss for logistic regression model:",metrics.log_loss(y_test,y_test_predict))


# In[ ]:




