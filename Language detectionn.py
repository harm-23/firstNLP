#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder


# In[10]:


data = pd.read_csv(r"Downloads\Language Detection.csv")
data.head(5)


# In[11]:


data["Language"].value_counts()


# In[12]:


X = data["Text"]
y = data["Language"]


# In[13]:


le = LabelEncoder()
y = le.fit_transform(y)


# In[14]:


# creating a list for appending the preprocessed text
data_list = []
# iterating through all the text
for text in X:
       # removing the symbols and numbers
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        # converting the text to lower case
        text = text.lower()
        # appending to data_list
        data_list.append(text)


# In[15]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
X.shape # (10337, 39419)


# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[17]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)


# In[18]:


y_pred = model.predict(x_test)


# In[19]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy is :",ac)
# Accuracy is : 


# In[20]:


plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()


# In[21]:


def predict(text):
     x = cv.transform([text]).toarray() # converting text to bag of words model (Vector)
     lang = model.predict(x) # predicting the language
     lang = le.inverse_transform(lang) # finding the language corresponding the the predicted value
     print("The langauge is in",lang[0]) # printing the language


# In[32]:


predict('ich bin ein mann')


# In[ ]:




