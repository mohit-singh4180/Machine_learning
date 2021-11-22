#!/usr/bin/env python
# coding: utf-8

# In[104]:


import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
            


# In[105]:


data = pd.read_csv("D:\AI\Salary_data.csv")


# In[106]:


data.info()


# In[107]:


### Split the data into x and y(y should be at the end,:-> all row 0 is first col, -1 is last col, :-1 all col except last)


# In[108]:


x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[109]:


y


# In[110]:


y.ndim


# In[111]:


x


# In[112]:


x.ndim


# In[113]:


###splittig data into training and testing data


# In[114]:


from sklearn.model_selection import train_test_split


# In[115]:



xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
### everytime stae will be same
train_test_split(x,y,test_size=0.3,random_state=123)


# In[116]:


xtrain


# In[117]:


xtest


# In[118]:


xtrain


# In[119]:


from sklearn.linear_model import LinearRegression


# In[120]:


model=LinearRegression()
model.fit(xtrain,ytrain)


# In[121]:


model.fit(xtrain,ytrain)


# In[122]:


model.predict(xtest)


# In[123]:


ytest


# In[124]:


model.score(xtest,ytest)


# In[ ]:





# In[ ]:





# In[125]:


model.predict(xtest)


# In[126]:


xtest


# In[127]:


### do normalization for x


# In[128]:


from sklearn.preprocessing import MinMaxScaler


# In[129]:


scaler=MinMaxScaler()


# In[130]:


####compare data with original values


# In[131]:


data2=pd.DataFrame(xtest,columns=["xtest"])


# In[132]:


data2.info()


# In[133]:


data2["y-pred"]=model.predict(xtest)
ypred=model.predict(xtest)


# In[134]:


data2["yi"]=ytest


# In[135]:


data2


# In[136]:


model.intercept_


# In[137]:


model.coef_


# In[138]:


ytest


# In[139]:


ypred


# In[140]:


from sklearn.metrics import mean_squared_error


# In[141]:


###use MSE if you want to check if you are goiong in right direction
   mean_squared_error(ytest,ypred)


# In[142]:


### Good way is to use score


# In[143]:


model.score(xtest,ytest)


# In[144]:


###Visualization


# In[152]:


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


### Predict value for one input


# In[155]:


model.predict([[15]])


# In[ ]:





# In[ ]:





# In[ ]:




