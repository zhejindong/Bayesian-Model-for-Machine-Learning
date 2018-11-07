
# coding: utf-8

# In[115]:


from numpy import *
import csv
import numpy as np
import math
from scipy.stats import norm
#import the data to Python 
with open('/Users/zhejindong/Desktop/movies_csv/ratings.csv') as csvfile:
    readCSV=csv.reader(csvfile)
    user=[]
    movie=[]
    rate=[]
    data=[]
    for row in readCSV:
        data.append(row)
        user.append(int(row[0]))
        movie.append(int(row[1]))
        rate.append(int(row[2]))


# In[116]:


total_user=np.unique(user).max()
total_movie=np.unique(movie).max()
d=5
c=1
v=1


# In[117]:


user_distribution=[]
movie_distribution=[]
for i in range(0,total_user+1):
    user_distribution.append(np.random.normal(0,0.1,d))
for j in range(0,total_movie+1):
    movie_distribution.append(np.random.normal(0,0.1,d))


# In[118]:


# initialize the two matrix 
U=np.mat(user_distribution).T
V=np.mat(movie_distribution).T


# In[119]:


user=np.array(user)[:,np.newaxis]
movie=np.array(movie)[:,np.newaxis]
rate=np.array(rate)[:,np.newaxis]
train_data=np.concatenate((user,movie,rate),axis=1)


# In[120]:


# EM 
import timeit
start = timeit.default_timer()
Eq=np.mat(np.full((total_user+1,total_movie+1),-100.0))
likelihood4=[]
for iteration in range(1,101):
    temp0=(U.T*V)/v
    temp1=norm.cdf(-temp0)
    temp2=norm.pdf(-temp0)
    for i in train_data:
            if i[2]==1:
                Eq[i[0],i[1]]=temp0[i[0],i[1]]+v*temp2[i[0],i[1]]/(1-temp1[i[0],i[1]])
            else:
                Eq[i[0],i[1]]=temp0[i[0],i[1]]-v*temp2[i[0],i[1]]/temp1[i[0],i[1]]
            
    # M step 
    
    for i in range(1,total_user+1):
        index=np.argwhere(train_data[:,0]==i)[:,0]
        index_j=np.unique(train_data[index][:,1])
        p1=((1.0/c)*np.eye(5)+1.0/v*np.dot(V[:,index_j],V[:,index_j].T)).T.I
        p2=np.dot(V[:,index_j],Eq[i,index_j].T)
        U[:,i]=(1.0/v)*np.dot(p1,p2)
    
    for j in range(1,total_movie+1):
        index=np.argwhere(train_data[:,1]==j)[:,0]
        index_i=np.unique(train_data[index][:,0])
        p1=((1.0/c)*np.eye(5)+1.0/v*np.dot(U[:,index_i],U[:,index_i].T)).T.I
        p2=np.dot(U[:,index_i],Eq[index_i,j])
        V[:,j]=(1.0/v)*np.dot(p1,p2)
        
    # calculate likelihood 
    Sum=0
    temp0=(U.T*V)/v
    temp1=np.log(norm.cdf(temp0))
    temp2=np.log(1-norm.cdf(temp0))
    for i in train_data: 
            if i[2]==1:
                Sum+=temp1[i[0],i[1]]*((1+i[2])/2)
            else:
                Sum+=((1-i[2])/2)*temp2[i[0],i[1]]
                
    Sum1=0
    for i in range(1,U.shape[1]):
            Sum1+=float(np.dot(U[:,i].T,U[:,i]))
        
    Sum2=0
    for j in range(1,U.shape[1]):
            Sum2+=float(np.dot(V[:,j].T,V[:,j]))
    Sum=Sum-1.0/(2*c)*(Sum1+Sum2)-(total_user+total_movie)*d/2*math.log(2*c*pi)
    
    #print iteration 
   #print Sum
    likelihood4.append(Sum)
    
stop = timeit.default_timer()
print('Time: ', stop - start)  


# In[48]:


import matplotlib.pyplot as plt
y=list(np.arange(2,100))
plt.scatter(y,likelihood[2:101])
plt.xlabel('iteration')
plt.ylabel('P(ln(R,U,V))')
plt.savefig('Desktop/qer.jpg')


# In[132]:


y=list(np.arange(20,100))
plt.plot(y,likelihood[20:101],'bo',y,likelihood1[20:101],'r^',y,likelihood2[20:101],'gs',y,likelihood3[20:101],'y*',y,likelihood4[20:101],'k*')
plt.xlabel('iteration')
plt.ylabel('P(ln(R,U,V))')
plt.savefig('Desktop/qerr.jpg')


# In[67]:


with open('/Users/zhejindong/Desktop/movies_csv/ratings_test.csv') as csvfile:
    readCSV=csv.reader(csvfile)
    user_test=[]
    movie_test=[]
    rate_test=[]
    data_test=[]
    for row in readCSV:
        data_test.append(row)
        user_test.append(int(row[0]))
        movie_test.append(int(row[1]))
        rate_test.append(int(row[2]))


# In[107]:


# predict 
a=norm.cdf(U.T*V/v)
predict=[]
for i in range(len(user_test)):
    predict.append(a[user_test[i],movie_test[i]])


# In[108]:


for i in range(len(user_test)):
    if predict[i]>=0.5:
        predict[i]=1        
    else:
        predict[i]=-1


# In[109]:


right1=0
wrong1=0
right0=0
wrong0=0
for i in range(len(user_test)):
    if rate_test[i]==1:
        if predict[i]==rate_test[i]:
            right1+=1
        else:
            wrong1+=1
    if rate_test[i]==-1:
        if predict[i]==rate_test[i]:
            right0+=1
        else:
            wrong0+=1


# In[110]:


right1


# In[111]:


wrong1


# In[112]:


right0


# In[113]:


wrong0

