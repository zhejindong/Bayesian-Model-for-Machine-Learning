
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# Import data 
train_x=pd.read_csv("hw1-data/X_train.csv",header=None)
train_y=pd.read_csv("hw1-data/Y_train.csv",header=None)
test_x=pd.read_csv("hw1-data/X_test.csv",header=None)
test_y=pd.read_csv("hw1-data/Y_test.csv",header=None)


# In[2]:


# Scale the data standadize normalizatoin ???? 
# we do not touch 1 because this is design matrix
train_x.loc[:,:5]=(train_x.loc[:,:5]-train_x.loc[:,:5].mean())/train_x.loc[:,:5].std()
test_x.loc[:,:5]=(test_x.loc[:,:5]-test_x.loc[:,:5].mean())/test_x.loc[:,:5].std()


# In[3]:


# calculate the svd of train_X matrix 
(u, s, vh)=np.linalg.svd(train_x.values,full_matrices=False)
# calculate the df(lamda)


# In[4]:


ans=[sum(np.square(s)/(np.square(s)))]
weight=np.linalg.inv(np.dot(train_x.transpose(),train_x)).dot(train_x.transpose()).dot(train_y)
for lamda in range(1,5000):
    ans.append(sum(np.square(s)/(np.square(s)+lamda)))
    weight=np.hstack((weight,np.linalg.inv(np.dot(train_x.transpose(),train_x)+lamda*np.identity(7)).dot(train_x.transpose()).dot(train_y)))  
    


# In[5]:


plt.figure(figsize=(10,10))
line1,=plt.plot(ans,weight[0,:],'g')
line2,=plt.plot(ans,weight[1,:],'b')
line3,=plt.plot(ans,weight[2,:],'r')
line4,=plt.plot(ans,weight[3,:],'orange')
line5,=plt.plot(ans,weight[4,:],'brown')
line6,=plt.plot(ans,weight[5,:],'y')
line7,=plt.plot(ans,weight[6,:],'darkviolet')
plt.gca().legend((line1,line2,line3,line4,line5,line6),("cylinders","displacement","horsepower","weight","acceleration","year made"))
plt.ylabel('weight')
plt.xlabel('df(lamda)')
plt.show()  


# In[6]:


#Forλ= 0,...,50, predict all 42 test cases. Plot the root mean squared error (RMSE)3on the testset as a function ofλ—notas a function ofdf(λ). What does this figure tell you when choosingλfor this problem (and when choosing between ridge regression and least squares)?

def RMSE(y_predict,y_true):
    return np.sqrt(np.mean(np.square(y_predict-y_true),axis=0))


# In[8]:


y_predict=np.dot(test_x,weight[:,:51])
#y_predict-test_y.values
R1=RMSE(y_predict,test_y.values)
# plot RMSE OF Lamda 
plt.figure(figsize=(10,10))
plt.ylabel('RMSE')
plt.xlabel('Lambda')
plt.plot(range(0,51),R1,'-go')


# ## Part 2

# In[9]:


# Import data 
train_x=pd.read_csv("hw1-data/X_train.csv",header=None)
train_y=pd.read_csv("hw1-data/Y_train.csv",header=None)
test_x=pd.read_csv("hw1-data/X_test.csv",header=None)
test_y=pd.read_csv("hw1-data/Y_test.csv",header=None)


# In[10]:


# order=1

y_predict=np.dot(test_x,weight[:,:101])
#y_predict-test_y.values
R1=RMSE(y_predict,test_y.values)




# order=2
# add higher order features 
for i in range(6,12):
    train_x.loc[:,i]=train_x.loc[:,i-6]*train_x.loc[:,i-6]
    test_x.loc[:,i]=test_x.loc[:,i-6]*test_x.loc[:,i-6]
# scale the data 
train_x=(train_x-train_x.mean())/train_x.std()
test_x=(test_x-test_x.mean())/test_x.std()
train_x.loc[:,12]=1
test_x.loc[:,12]=1

weight2=np.linalg.inv(np.dot(train_x.transpose(),train_x)).dot(train_x.transpose()).dot(train_y)
for lamda in range(1,101):
    weight2=np.hstack((weight2,np.linalg.inv(np.dot(train_x.transpose(),train_x)+lamda*np.identity(train_x.shape[1])).dot(train_x.transpose()).dot(train_y)))  
    
y_predict=np.dot(test_x,weight2)
#y_predict-test_y.values
R2=RMSE(y_predict,test_y.values)
# plot RMSE OF Lamda 
#plt.figure(figsize=(10,10))
#plt.plot(range(0,101),R2,'go')


# In[11]:


# order=3
# add higher order features 
for i in range(12,18):
    train_x.loc[:,i]=train_x.loc[:,i-6]*train_x.loc[:,i-6]
    test_x.loc[:,i]=test_x.loc[:,i-6]*test_x.loc[:,i-6]
# scale the data 
train_x=(train_x-train_x.mean())/train_x.std()
test_x=(test_x-test_x.mean())/test_x.std()
train_x.loc[:,18]=1
test_x.loc[:,18]=1

weight3=np.linalg.inv(np.dot(train_x.transpose(),train_x)).dot(train_x.transpose()).dot(train_y)
for lamda in range(1,101):
    weight3=np.hstack((weight3,np.linalg.inv(np.dot(train_x.transpose(),train_x)+lamda*np.identity(train_x.shape[1])).dot(train_x.transpose()).dot(train_y)))  
    
y_predict=np.dot(test_x,weight3)
#y_predict-test_y.values
R3=RMSE(y_predict,test_y.values)
# plot RMSE OF Lamda 
#plt.figure(figsize=(10,10))
#plt.plot(range(0,101),R3,'go')





# In[12]:


#plt.figure(figsize=(10,10))
#plt.plot(range(0,101),R3,'go')

plt.figure(figsize=(10,10))
line1,=plt.plot(range(101),R1,'-go')
line2,=plt.plot(range(101),R2,'-bo')
line3,=plt.plot(range(101),R3,'-ro')
plt.gca().legend((line1,line2,line3),("p=1","p=2","p=3"))
plt.ylabel('RMSE')
plt.xlabel('Lambda')
plt.show()  

