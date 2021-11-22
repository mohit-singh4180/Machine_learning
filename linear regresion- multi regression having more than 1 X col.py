#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
            


# In[6]:


data = pd.read_csv(r"D:\AI\21112021\02Students.csv")


# In[7]:


data.info()


# In[107]:


### Split the data into x and y(y should be at the end,:-> all row 0 is first col, -1 is last col, :-1 all col except last)


# In[10]:



data
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[109]:


y


# In[110]:


y.ndim


# In[11]:


x


# In[112]:


x.ndim


# In[113]:


###splittig data into training and testing data


# In[55]:


from sklearn.model_selection import train_test_split


# In[35]:



xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
### everytime stae will be same
train_test_split(x,y,test_size=0.3,random_state=123)


# In[16]:


xtrain


# In[56]:


xtest


# In[57]:


xtrain


# In[58]:


from sklearn.linear_model import LinearRegression


# In[59]:


model=LinearRegression()
model.fit(xtrain,ytrain)


# In[60]:


model.fit(xtrain,ytrain)


# In[61]:


model.predict(xtest)


# In[62]:


ytest


# In[74]:


model.score(xtrain,ytrain)


# In[73]:



model.score(xtest,ytest)


# In[64]:


model.intercept_


# In[ ]:





# In[65]:


model.predict(xtest)


# In[51]:


xtest


# In[127]:


### do normalization for x


# In[52]:


from sklearn.preprocessing import MinMaxScaler


# In[53]:


scaler=MinMaxScaler()


# In[46]:


####compare data with original values


# In[54]:


data2=pd.DataFrame(xtest,columns=["xtest"])


# In[132]:


data2.info()


# In[48]:


data2["y-pred"]=model.predict(xtest)
ypred=model.predict(xtest)


# In[134]:


data2["yi"]=ytest


# In[135]:


data2


# In[136]:


model.intercept_


# In[25]:


model.coef_


# In[138]:


ytest


# In[26]:


ypred


# In[66]:


from sklearn.metrics import mean_squared_error


# In[67]:


###use MSE if you want to check if you are goiong in right direction
   mean_squared_error(ytest,ypred)


# In[142]:


### Good way is to use score


# In[68]:


model.score(xtest,ytest)


# In[144]:


###Visualization


# In[69]:


### Visualization for training data
plt.scatter(xtrain,ytrain,c="r")

### plot graph fr xtrain and ypred
plt.plot(xtrain,model.predict(xtrain))
plt.title("EXP vs salary(Training data)")
plt.xlabel("EXP xtrain")
plt.ylabel("salary ypred")
plt.show()


# In[ ]:





# In[153]:


plt.scatter(xtest,ytest,c="black")
### Test with training data
plt.plot(xtrain,model.predict(xtrain))
plt.title("EXP vs salary(Test data)")
plt.xlabel("EXP xtest")
plt.ylabel("salary ytest")
plt.show()


# In[ ]:


### Predict value for one input, value should always be in 2 D array


# In[155]:


model.predict([[15]])


# In[ ]:





# In[ ]:





# In[ ]:




