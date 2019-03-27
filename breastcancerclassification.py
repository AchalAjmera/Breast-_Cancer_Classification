
# coding: utf-8

# In[1]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 
# 

# In[2]:


# importing the dataset 
from sklearn.datasets import load_breast_cancer 


# In[3]:


# loading the dataset 
cancer_data = load_breast_cancer()
data = pd.read_csv('data.csv')
data[:20]


# In[4]:


cancer_data.keys()


# In[5]:


# Description
print(cancer_data ['DESCR']);


# In[6]:


# Target
print('Here 0 and 1 are Malignant and Benign')
cancer_data['target']


# In[7]:


# Target names
print(cancer_data['target_names'])


# In[8]:


# Features names
print(cancer_data['feature_names'])


# In[9]:


# Instances and Attributes
cancer_data['data'].shape


# In[10]:


# Organize our data 
label_names =cancer_data['target_names'] 
labels = cancer_data['target'] 
feature_names = cancer_data['feature_names'] 
features = cancer_data['data'] 


# In[11]:


#Creating Train test split
from sklearn.model_selection import train_test_split 
train, test, train_labels, test_labels = train_test_split(features, labels,test_size = 0.2, random_state =0)


# In[12]:


# Learning model 
from sklearn.tree import DecisionTreeClassifier  
tree = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0) 
# training the classifier 
model = tree.fit(train, train_labels)


# In[13]:


# making the predictions 
predictions = tree.predict(test) 


# In[14]:


# printing the predictions 
print(predictions) 


# In[15]:


# Accuracy measuring function 
from sklearn.metrics import accuracy_score 
print(accuracy_score(test_labels, predictions)) 


# In[16]:


n=11
mi=np.zeros((n))
for i in range(1,n):
    tree= DecisionTreeClassifier(criterion='entropy',max_depth=i,random_state=0)
    model = tree.fit(train, train_labels)
    predictions = tree.predict(test)
   ## print(predictions)
    print("")
    mi[i]=accuracy_score(test_labels, predictions)
    print('Accuracy score for max_depth:',i , mi[i],)
    print("")


# In[17]:


print("Accuracy score",max(mi),'at max_depth',mi.argmax())

