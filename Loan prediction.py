#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


train_df = pd.read_csv(r'C:\Users\vividha\Desktop\Datasets\Loan Prediction train.csv')
train_df.head()


# In[3]:


train_df = train_df.drop('Loan_ID', axis = 1)
train_df.head()


# In[4]:


total = train_df.isnull().sum().sort_values(ascending=False)
total


# In[5]:


percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
percent


# In[6]:


missing_values = pd.concat([total, percent], keys = ('Total', 'Percent'))
missing_values


# In[7]:


defaultint = 0
defaultstr = 'Nan'
defaultfloat = 0.0

features = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome','Credit_History','LoanAmount', 'Loan_Amount_Term','Gender', 'Married', 'Education', 'Property_Area', 'Loan_Status']

int_features = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome','Credit_History']
float_features = ['LoanAmount', 'Loan_Amount_Term']
str_features = ['Gender', 'Married', 'Education', 'Property_Area', 'Loan_Status']

for features in train_df:
    if features in int_features:
        train_df[features] = train_df[features].fillna(defaultint)
    elif features in float_features:
        train_df[features] =  train_df[features].fillna(defaultfloat)
    else:
        train_df[features] =  train_df[features].fillna(defaultstr)
        


# In[8]:


total = train_df.isnull().sum().sort_values(ascending = False)
total


# In[9]:


train_df['Dependents'] = train_df['Dependents'].replace(0, '0')


# In[10]:


print(train_df['Dependents'].unique())


# In[11]:


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

le_dep = preprocessing.LabelEncoder()
le_dep.fit(train_df['Dependents'].unique())
train_df['Dependents'] = le_dep.transform(train_df['Dependents'])
                                          
le_sex = preprocessing.LabelEncoder()
le_sex.fit(train_df['Gender'].unique())
train_df['Gender'] = le_sex.transform(train_df['Gender'])

le_married = preprocessing.LabelEncoder()
le_married.fit(train_df['Married'].unique())
train_df['Married'] = le_married.transform(train_df['Married'])

le_edu = preprocessing.LabelEncoder()
le_edu.fit(train_df['Education'].unique())
train_df['Education'] = le_edu.transform(train_df['Education'])

le_self = preprocessing.LabelEncoder()
le_self.fit(train_df['Self_Employed'].unique())
train_df['Self_Employed'] = le_self.transform(train_df['Self_Employed'])

le_pro = preprocessing.LabelEncoder()
le_pro.fit(train_df['Property_Area'].unique())
train_df['Property_Area'] = le_pro.transform(train_df['Property_Area'])

le_sta = preprocessing.LabelEncoder()
le_sta.fit(train_df['Loan_Status'].unique())
train_df['Loan_Status'] = le_sta.transform(train_df['Loan_Status'])


# In[12]:


X = train_df.drop('Loan_Status', axis = 1)
Y = train_df['Loan_Status']


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size  = 0.2)


# In[14]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train, y_train)


# In[18]:


y_pred = log.predict(x_test)


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print(confusion_matrix(y_test, y_pred, labels=[1,0]))


# In[24]:


cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
np.set_printoptions(precision=2)


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')


# In[25]:


test_df = pd.read_csv(r'C:\Users\vividha\Desktop\Datasets\Loan Prediction test.csv')
defaultint = 0
defaultstr = 'Nan'
defaultfloat = 0.0

features = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome','Credit_History','LoanAmount', 'Loan_Amount_Term','Gender', 'Married', 'Education', 'Property_Area']

int_features = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome','Credit_History']
float_features = ['LoanAmount', 'Loan_Amount_Term']
str_features = ['Gender', 'Married', 'Education', 'Property_Area']

for features in test_df:
    if features in int_features:
        test_df[features] = test_df[features].fillna(defaultint)
    elif features in float_features:
        test_df[features] =  test_df[features].fillna(defaultfloat)
    else:
        test_df[features] =  test_df[features].fillna(defaultstr)
        
        
test_df['Dependents'] = test_df['Dependents'].replace(0, '0')
        
le_dep = preprocessing.LabelEncoder()
le_dep.fit(test_df['Dependents'].unique())
test_df['Dependents'] = le_dep.transform(test_df['Dependents'])
                                          
le_sex = preprocessing.LabelEncoder()
le_sex.fit(test_df['Gender'].unique())
test_df['Gender'] = le_sex.transform(test_df['Gender'])

le_married = preprocessing.LabelEncoder()
le_married.fit(test_df['Married'].unique())
test_df['Married'] = le_married.transform(test_df['Married'])

le_edu = preprocessing.LabelEncoder()
le_edu.fit(test_df['Education'].unique())
test_df['Education'] = le_edu.transform(test_df['Education'])

le_self = preprocessing.LabelEncoder()
le_self.fit(test_df['Self_Employed'].unique())
test_df['Self_Employed'] = le_self.transform(test_df['Self_Employed'])

le_pro = preprocessing.LabelEncoder()
le_pro.fit(test_df['Property_Area'].unique())
test_df['Property_Area'] = le_pro.transform(test_df['Property_Area'])
   


# In[26]:


x_test1=test_df.drop('Loan_ID', axis=1)
y_test1=log.predict(x_test1)

