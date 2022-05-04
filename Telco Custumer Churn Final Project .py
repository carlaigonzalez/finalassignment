#!/usr/bin/env python
# coding: utf-8

# # Final Project: Telco Custumer Churn Predicting Model
# ## Carla Gonzalez, Ellis Brown, Anandhu Mahesh
# ### Our goal for this assignment is to be able to predict if a customer is in high risk of churn. 

# In[70]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("Telco-Customer-Churn.csv")


# ### As a first step is is always important to get familiarize with the dataset, we can do this by looking at its head, total lenght, value counts etc:

# In[71]:


df.head()


# In[72]:


print(len(df))


# In[73]:


df["Churn"].value_counts()


# In[74]:


df["MonthlyCharges"].describe()


# In[75]:


df["Contract"].value_counts()


# In[76]:


type_counts = df["Partner"].value_counts()
plt.rcParams["figure.figsize"] = (10,5)
type_counts.plot.bar()
plt.xlabel("Partner")
plt.ylabel("Number of Observations")


# In[77]:


df[df["Churn"]!= "No"]


# In[78]:


df["Contract"].value_counts()


# ### After getting to know the data we noticied people with month to month contracts are more likely to churn rather than one year, also there is a pattern in gender and partner status. With this two discoveries we decided the two hypothesis we want to test are:  1) There is a relationship between customers churning and the type of contract they have. 2) Customers are more likely to churn if they are not partnered and are female.
# 

# In[79]:


df['is_female'] = df['gender'].apply(lambda g: int(g == 'Female'))
df['has_partner'] = df['Partner'].apply(lambda p: int(p == 'Yes'))
df['is_month_to_month'] = df['Contract'].apply(lambda g: int(g == 'Month-to-month'))
df['is_one_year'] = df['Contract'].apply(lambda g: int(g == 'One year'))
df['is_two_year'] = df['Contract'].apply(lambda g: int(g == 'Two year'))


# In[80]:


from sklearn.metrics import accuracy_score

def train_logistic_regression(X, y):
    
    #Get number of examples
    N_EXAMPLES = len(y)
    TEST_SIZE = round(0.25 * N_EXAMPLES)
    
    # Split the data into training/testing sets
    X_train = X[:-TEST_SIZE]
    X_test = X[-TEST_SIZE:]

    # Split the targets into training/testing sets
    y_train = y[:-TEST_SIZE]
    y_test = y[-TEST_SIZE:]

    # Create linear regression object
    regr = linear_model.LogisticRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)
    
    print(accuracy_score(y_test, y_pred))
    
    return regr


# In[81]:


X = df[['is_female', 'has_partner', 'is_month_to_month', 'is_one_year', 'is_two_year']]
y = df['Churn']


# In[82]:


r = train_logistic_regression(X, y)
r.coef_


# ### With the use of Logistic Regression we can conclude 1) people with month-to-month contract have e^(1.6554457) times the odd of churn rather than one year contrtact customers who are e^(-.041919) less likely to churn, and two year contract customers who are e^(-1.61333368) less likely to churn. As we can see, it is much probable customers stay as customers if they contract their services for two years. And 2) Females have e^(0.50023097) times the odd of churn rather than men, in the case of partner status,  people with partners are e^(-.18145626) less likely to churn rather than single customers.
