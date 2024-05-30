#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


raw_mail_data= pd.read_csv('mail_data.csv')


# In[3]:


print(raw_mail_data)


# In[4]:


mail_data= raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[5]:


mail_data.head()


# In[6]:


mail_data.shape


# In[7]:


mail_data.loc[mail_data['Category'] == 'ham', 'Category',]=1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0


# In[8]:


X = mail_data['Message']

Y = mail_data['Category']


# In[9]:


print(X)


# In[10]:


print(Y)


# In[11]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example data (you should replace this with your actual dataset)
data = {'text': [
    "This is a sample sentence.", "Another example sentence.", "Machine learning is fun!",
    "Text data requires preprocessing.", "This is a new sentence.", "Another one here.",
    "Different words entirely.", "More text data.", "Some other example.", "Last sentence here."
], 'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Split the data into training and test sets
X = df['text']
Y = df['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the TfidfVectorizer
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Fit and transform the training data
X_train_features = feature_extraction.fit_transform(X_train)

# Transform the test data (using the same fitted vectorizer)
X_test_features = feature_extraction.transform(X_test)

# Ensure the labels are of integer type
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Debugging: Print the shapes of the feature matrices
print(f"Shape of X_train_features: {X_train_features.shape}")
print(f"Shape of X_test_features: {X_test_features.shape}")

# Define and train the model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Make predictions on the test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print(f"Test Accuracy: {accuracy_on_test_data}")


# In[12]:


print(X_train)


# In[13]:


print(X_train_features)


# In[14]:


model = LogisticRegression()
model.fit(X_train_features, Y_train)


# In[15]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[16]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[17]:


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[18]:


print('Accuracy on test data : ', accuracy_on_test_data)


# In[19]:


input_mail=["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]
input_data_features=feature_extraction.transform(input_mail)
prediction=model.predict(input_data_features)
print(prediction)
if(prediction[0]==1):
    print('Ham mail')
else:
    print('Spam mail')


# In[ ]:




