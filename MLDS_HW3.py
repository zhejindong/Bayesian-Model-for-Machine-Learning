
# coding: utf-8

# ## Problem 1 (K-means)
# 
# Implement the K-means algorithm discussed in class. Generate 500 observations from a mixture of three Gaussians on R2 with mixing weights π= [0.2,0.5,0.3] and means μ and covariances Σ

# In[1]:


# Generate data
from numpy.linalg import inv
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial import distance
elements = [1, 2, 3]
pi = [0.2, 0.5, 0.3]
x=np.random.choice(elements,500, p=pi)
u1=np.array([0,0])
u2=np.array([3,0])
u3=np.array([0,3])
cov1=np.array([[1,0],[0,1]])
cov2=np.array([[1,0],[0,1]])
cov3=np.array([[1,0],[0,1]])


# In[2]:


a=np.random.multivariate_normal(u1,cov1,(x==1).sum())
b=np.random.multivariate_normal(u2,cov2,(x==2).sum())
c=np.random.multivariate_normal(u3,cov3,(x==3).sum())
data=np.vstack((a,b,c))


# In[3]:


plt.scatter(data[:,0],data[:,1])


# ### a)
# For K= 2,3,4,5, plot the value of the K-means objective function per iteration for 20 iterations (the algorithm may converge before that).

# In[4]:


# Euclidean Distance Caculator
from copy import deepcopy
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
X=data
run_error=[]
for k in range(2,6):
    # random initialize centroids 
    C=np.random.random_sample((k, 2))*3
    C_old = np.zeros(C.shape)
    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(data))
    # Error func. - Distance between new centroids and old centroids
    error = dist(C, C_old, None)
    # Loop will run till the error becomes zero
    inner_error=[]
    for iteration in range(1,21):
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(data[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = deepcopy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(data)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)
        inner_error.append(error)
    run_error.append(inner_error)


# In[5]:


plt.plot(range(1,21),np.array(run_error).T)
plt.title('Objective function K means')
plt.show


# ### b)  
# For K= 3,5, plot the 500 data points and indicate the cluster of each for the final iteration bymarking it with a color or a symbol.

# In[6]:


# Euclidean Distance Caculator
from copy import deepcopy
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
X=data
run_error=[]
for k in [3,5]:
    # random initialize centroids 
    C=np.random.random_sample((k, 2))*3
    C_old = np.zeros(C.shape)
    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(data))
    # Error func. - Distance between new centroids and old centroids
    error = dist(C, C_old, None)
    # Loop will run till the error becomes zero
    inner_error=[]
    for iteration in range(1,21):
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(data[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = deepcopy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(data)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)
        inner_error.append(error)
        #print(error)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
    ax.set_title("K is {}".format(k))


# # Question 2

# In[7]:


import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#read in data 
train_x2=pd.read_csv('/Users/zhejindong/Desktop/hw3-data/Prob2_Xtrain.csv',header=None)
train_y2=pd.read_csv('/Users/zhejindong/Desktop/hw3-data/Prob2_ytrain.csv',header=None)
test_x2=pd.read_csv('/Users/zhejindong/Desktop/hw3-data/Prob2_Xtest.csv',header=None)
test_y2=pd.read_csv('/Users/zhejindong/Desktop/hw3-data/Prob2_ytest.csv',header=None)
train_x2['y']=train_y2
print(test_x2.shape)
print(train_x2.shape)
print(test_y2.shape)


# In[8]:


train_x2_0=train_x2.iloc[:,:][train_x2['y']==0]
train_x2_1=train_x2.iloc[:,:][train_x2['y']==1]


# In[9]:


train_x2_0.shape[0]


# In[10]:


train_x2_1.shape


# In[11]:


prior_0=train_x2_0.shape[0]/(train_x2_0.shape[0]+train_x2_1.shape[0])
prior_1=1-prior_0


# In this problem, you will implement the EM algorithm for the Gaussian mixture model, with the purposeof using it in a Bayes classifier. The data is a processed version of the spam email data you looked at in Homework 2. Now, each labeled pair(x,y) has x∈R10. We discussed how the Bayes classifier learnsclass-conditional densities, and unsupervised learning algorithms can be useful here.  In this problem,the class conditional density will be the Gaussian mixture model (GMM). In these experiments, please initialize  all  covariance  matrices  to  the  empirical  covariance  of  the  data  being  modeled. Randomlyinitialize the means by sampling from a single multivariate Gaussian where the parameters are the mean and covariance of the data being modeled. Initialize the mixing weights to be uniform.

# ### 3GMM on label 0

# In[12]:


from scipy import stats
from scipy.stats import multivariate_normal
run_res_0=[]
run_para_0=[]
for run in range(1,11):
    #print(run)
    c=3
    pi1=1.0/3
    pi2=1.0/3
    pi3=1.0/3
    mean=np.array(train_x2_0.iloc[:,:-1].mean())
    cov=np.cov(train_x2_0.iloc[:,:-1].T)
    u=np.random.multivariate_normal(mean,cov,3)
    u1=u[0]
    u2=u[1]
    u3=u[2]
    cov1=cov
    cov2=cov
    cov3=cov
    iter_res_0=[]
    iter_para_0=[]
    for iteration in range(1,31):
        #print(iteration)
        # E step
        x0=np.array(train_x2_0.iloc[:,:-1])
        p1=multivariate_normal.pdf(x0, mean=u1,cov=cov1)
        #print("p1:"+str(p1))
        p2=multivariate_normal.pdf(x0, mean=u2,cov=cov2)
        #print("p2:"+str(p2))
        p3=multivariate_normal.pdf(x0, mean=u3,cov=cov3)
        phi1=(p1*pi1/(p1*pi1+p2*pi2+p3*pi3)) # the proba that each data fall in cluster 1 2019,0
        phi2=(p2*pi2/(p1*pi1+p2*pi2+p3*pi3)) # the proba that each data fall in cluster 2
        phi3=(p3*pi3/(p1*pi1+p2*pi2+p3*pi3)) # the proba that each data fall in cluster 3
        of=np.log(pi1*p1+pi2*p2+pi3*p3).sum()
        #print(pi1,pi2,pi3)
        iter_para_0.append({"u1":u1,"u2":u2,"u3":u3,"cov1":cov1,"cov2":cov2,"cov3":cov3,"pi1":pi1,"pi2":pi2,"pi3":pi3})
        iter_res_0.append(of)
        # M step
        n=phi1.shape[0]
        n1=phi1.sum()
        n2=phi2.sum()
        n3=phi3.sum()
        # Update Pi
        pi1=n1/n
        pi2=n2/n
        pi3=n3/n
        # Update uk  
        u1=(phi1.reshape(-1,+1)*x0).sum(axis=0)/n1
        u2=(phi2.reshape(-1,+1)*x0).sum(axis=0)/n2
        u3=(phi3.reshape(-1,+1)*x0).sum(axis=0)/n3
        # update cov
        cov1=(phi1.reshape(-1,+1)*(x0-u1)).T.dot((x0-u1))/n1
        cov2=(phi2.reshape(-1,+1)*(x0-u2)).T.dot((x0-u2))/n2
        cov3=(phi3.reshape(-1,+1)*(x0-u3)).T.dot((x0-u3))/n3   
        
        #iter_para.append((u1,u2,u3,cov1,cov2,cov3,p1,p2,p3))
    run_res_0.append(iter_res_0)
    run_para_0.append(iter_para_0)

# plot 
import matplotlib.pyplot as plt
plt.plot(range(5,30),np.array(run_res_0).T[5:])
plt.title('Objective function 0 class')
plt.show


# ### 3GMM on label 1

# In[13]:


from scipy import stats
from scipy.stats import multivariate_normal
run_res=[]
run_para=[]
for run in range(1,11):
    #print(run)
    c=3
    pi1=1.0/3
    pi2=1.0/3
    pi3=1.0/3
    mean=np.array(train_x2_1.iloc[:,:-1].mean())
    cov=np.cov(train_x2_1.iloc[:,:-1].T)
    u=np.random.multivariate_normal(mean,cov,3)
    u1=u[0]
    u2=u[1]
    u3=u[2]
    cov1=cov
    cov2=cov
    cov3=cov
    iter_res=[]
    iter_para=[]
    for iteration in range(1,31):
        #print(iteration)
        # E step
        x0=np.array(train_x2_1.iloc[:,:-1])
        p1=multivariate_normal.pdf(x0, mean=u1,cov=cov1)
        p2=multivariate_normal.pdf(x0, mean=u2,cov=cov2)
        p3=multivariate_normal.pdf(x0, mean=u3,cov=cov3)
        phi1=(p1*pi1/(p1*pi1+p2*pi2+p3*pi3)) # the proba that each data fall in cluster 1 2019,0
        phi2=(p2*pi2/(p1*pi1+p2*pi2+p3*pi3)) # the proba that each data fall in cluster 2
        phi3=(p3*pi3/(p1*pi1+p2*pi2+p3*pi3)) # the proba that each data fall in cluster 3
        of=np.log(pi1*p1+pi2*p2+pi3*p3).sum()
        iter_para.append({"u1":u1,"u2":u2,"u3":u3,"cov1":cov1,"cov2":cov2,"cov3":cov3,"pi1":pi1,"pi2":pi2,"pi3":pi3})
        iter_res.append(of)
        # M step
        n=phi1.shape[0]
        n1=phi1.sum()
        n2=phi2.sum()
        n3=phi3.sum()
        # Update Pi
        pi1=n1/n
        pi2=n2/n
        pi3=n3/n
        # Update uk  
        u1=(phi1.reshape(-1,+1)*x0).sum(axis=0)/n1
        u2=(phi2.reshape(-1,+1)*x0).sum(axis=0)/n2
        u3=(phi3.reshape(-1,+1)*x0).sum(axis=0)/n3
        # update cov
        cov1=(phi1.reshape(-1,+1)*(x0-u1)).T.dot((x0-u1))/n1
        cov2=(phi2.reshape(-1,+1)*(x0-u2)).T.dot((x0-u2))/n2
        cov3=(phi3.reshape(-1,+1)*(x0-u3)).T.dot((x0-u3))/n3   
    
    run_res.append(iter_res)
    run_para.append(iter_para)


# In[14]:


# plot 
import matplotlib.pyplot as plt
plt.plot(range(5,30),np.array(run_res).T[5:])
plt.title('Objective function 1 class')
plt.show


# ### b
# Using the best run for each class after 30 iterations, predict the testing data using a Bayes classifierand show the result in a2×2confusion matrix,  along with the accuracy percentage. Repeat this process for a 1-, 2-, 3- and 4-Gaussian mixture model.Show all results nearby each other,and don’t repeat Part (a) for these other cases.Note that a 1-Gaussian GMM doesn’t require an algorithm, although your implementation will likely still work in this case.

# In[15]:


max_run_0=np.argmax(np.array(run_res_0).T[:][-1])
max_run_1=np.argmax(np.array(run_res).T[:][-1])

#proba0=run_para_0[max_run_0][-1]
x0=test_x2
p1=multivariate_normal.pdf(x0, mean=run_para_0[max_run_0][-1]['u1'],cov=run_para_0[max_run_0][-1]['cov1'])
p2=multivariate_normal.pdf(x0, mean=run_para_0[max_run_0][-1]['u2'],cov=run_para_0[max_run_0][-1]['cov2'])
p3=multivariate_normal.pdf(x0, mean=run_para_0[max_run_0][-1]['u3'],cov=run_para_0[max_run_0][-1]['cov3'])

of_0=np.log(pi1*p1+pi2*p2+pi3*p3)+np.log(prior_0)
# make a prediction 
#proba0=run_para_0[max_run_0][-1]
x0=test_x2
p1=multivariate_normal.pdf(x0, mean=run_para[max_run_1][-1]['u1'],cov=run_para[max_run_1][-1]['cov1'])
p2=multivariate_normal.pdf(x0, mean=run_para[max_run_1][-1]['u2'],cov=run_para[max_run_1][-1]['cov2'])
p3=multivariate_normal.pdf(x0, mean=run_para[max_run_1][-1]['u3'],cov=run_para[max_run_1][-1]['cov3'])

of_1=np.log(pi1*p1+pi2*p2+pi3*p3)+np.log(prior_1)

prediction=of_1>of_0


# #####  Confusion Matrix of 3 GMM prediction 

# In[16]:


from sklearn.metrics import confusion_matrix
matrix3=(confusion_matrix(test_y2, prediction))
accuracy3=(matrix3[0][0]+matrix3[1][1])/matrix3.sum()


# ### 1GMM MODEL 

# In[50]:


# label 0
from scipy import stats
from scipy.stats import multivariate_normal
run_res_0=[]
run_para_0=[]
for run in range(1,11):
    #print(run)
    c=1
    pi1=1.0/1
    mean=np.array(train_x2_0.iloc[:,:-1].mean())
    cov=np.cov(train_x2_0.iloc[:,:-1].T)
    u=np.random.multivariate_normal(mean,cov,1)
    u1=u[0]
    #u3=u[2]
    cov1=cov
    iter_res_0=[]
    iter_para_0=[]
    for iteration in range(1,31):
        # E step
        x0=np.array(train_x2_0.iloc[:,:-1])
        p1=multivariate_normal.pdf(x0, mean=u1,cov=cov1)
        phi1=(p1*pi1/(p1*pi1)) # the proba that each data fall in cluster 1 2019,0
        of=np.log(pi1*p1).sum()
        iter_para_0.append({"u1":u1,"cov1":cov1,"pi1":pi1})
        iter_res_0.append(of)
        # M step
        n=phi1.shape[0]
        n1=phi1.sum()
        # Update Pi
        pi1=n1/n
        # Update uk  
        u1=(phi1.reshape(-1,+1)*x0).sum(axis=0)/n1
        # update cov
        cov1=(phi1.reshape(-1,+1)*(x0-u1)).T.dot((x0-u1))/n1
    run_res_0.append(iter_res_0)
    run_para_0.append(iter_para_0)

# label 1
run_res_1=[]
run_para_1=[]
for run in range(1,11):
    #print(run)
    c=1
    pi1=1.0/1
    mean=np.array(train_x2_1.iloc[:,:-1].mean())
    cov=np.cov(train_x2_1.iloc[:,:-1].T)
    u=np.random.multivariate_normal(mean,cov,1)
    u1=u[0]
    #u3=u[2]
    cov1=cov
    iter_res_1=[]
    iter_para_1=[]
    for iteration in range(1,31):
        # E step
        x0=np.array(train_x2_1.iloc[:,:-1])
        p1=multivariate_normal.pdf(x0, mean=u1,cov=cov1)
        phi1=(p1*pi1/(p1*pi1)) # the proba that each data fall in cluster 1 2019,0
        of=np.log(pi1*p1).sum()
        iter_para_1.append({"u1":u1,"cov1":cov1,"pi1":pi1})
        iter_res_1.append(of)
        # M step
        n=phi1.shape[0]
        n1=phi1.sum()
        # Update Pi
        pi1=n1/n
        # Update uk  
        u1=(phi1.reshape(-1,+1)*x0).sum(axis=0)/n1
        # update cov
        cov1=(phi1.reshape(-1,+1)*(x0-u1)).T.dot((x0-u1))/n1
    run_res_1.append(iter_res_1)
    run_para_1.append(iter_para_1)


# In[51]:


max_run_0=np.argmax(np.array(run_res_0).T[:][-1])
max_run_1=np.argmax(np.array(run_res_1).T[:][-1])
#proba0=run_para_0[max_run_0][-1]
x0=test_x2
p1=multivariate_normal.pdf(x0, mean=run_para_0[max_run_0][-1]['u1'],cov=run_para_0[max_run_0][-1]['cov1'])
#phi1=(p1*pi1/(p1*pi1)) # the proba that each data fall in cluster 1 2019,0
of_0=np.log(p1)+np.log(prior_0)
# make a prediction 
x0=test_x2
p1=multivariate_normal.pdf(x0, mean=run_para_1[max_run_1][-1]['u1'],cov=run_para_1[max_run_1][-1]['cov1'])
#phi1=(p1*pi1/(p1*pi1)) # the proba that each data fall in cluster 1 2019,0
of_1=np.log(p1)+np.log(prior_1)
prediction=of_1>of_0


# #####  Confusion Matrix of 1GMM 

# In[52]:


from sklearn.metrics import confusion_matrix
matrix1=(confusion_matrix(test_y2, prediction))
accuracy1=(matrix1[0][0]+matrix1[1][1])/matrix1.sum()


# ### 2GMM 

# In[46]:


# label 0
from scipy import stats
from scipy.stats import multivariate_normal
run_res_0=[]
run_para_0=[]
for run in range(1,11):
    #print(run)
    c=3
    pi1=1.0/2
    pi2=1.0/2
    #pi3=1.0/3
    mean=np.array(train_x2_0.iloc[:,:-1].mean())
    cov=np.cov(train_x2_0.iloc[:,:-1].T)
    u=np.random.multivariate_normal(mean,cov,2)
    u1=u[0]
    u2=u[1]
    #u3=u[2]
    cov1=cov
    cov2=cov
    #cov3=cov
    iter_res_0=[]
    iter_para_0=[]
    for iteration in range(1,31):
        #print(iteration)
        # E step
        x0=np.array(train_x2_0.iloc[:,:-1])
        p1=multivariate_normal.pdf(x0, mean=u1,cov=cov1)
        #print("p1:"+str(p1))
        p2=multivariate_normal.pdf(x0, mean=u2,cov=cov2)
        phi1=(p1*pi1/(p1*pi1+p2*pi2)) # the proba that each data fall in cluster 1 2019,0
        phi2=(p2*pi2/(p1*pi1+p2*pi2)) # the proba that each data fall in cluster 2
        of=np.log(pi1*p1+pi2*p2).sum()
        #print(pi1,pi2,pi3)
        iter_para_0.append({"u1":u1,"u2":u2,"cov1":cov1,"cov2":cov2,"pi1":pi1,"pi2":pi2})
        iter_res_0.append(of)
        # M step
        n=phi1.shape[0]
        n1=phi1.sum()
        n2=phi2.sum()
        #n3=phi3.sum()
        # Update Pi
        pi1=n1/n
        pi2=n2/n
        #pi3=n3/n
        # Update uk  
        u1=(phi1.reshape(-1,+1)*x0).sum(axis=0)/n1
        u2=(phi2.reshape(-1,+1)*x0).sum(axis=0)/n2
        #u3=(phi3.reshape(-1,+1)*x0).sum(axis=0)/n3
        # update cov
        cov1=(phi1.reshape(-1,+1)*(x0-u1)).T.dot((x0-u1))/n1
        cov2=(phi2.reshape(-1,+1)*(x0-u2)).T.dot((x0-u2))/n2
    run_res_0.append(iter_res_0)
    run_para_0.append(iter_para_0)
    
# label 1
from scipy import stats
from scipy.stats import multivariate_normal
run_res_1=[]
run_para_1=[]
for run in range(1,11):
    #print(run)
    c=3
    pi1=1.0/2
    pi2=1.0/2
    #pi3=1.0/3
    mean=np.array(train_x2_1.iloc[:,:-1].mean())
    cov=np.cov(train_x2_1.iloc[:,:-1].T)
    u=np.random.multivariate_normal(mean,cov,2)
    u1=u[0]
    u2=u[1]
    #u3=u[2]
    cov1=cov
    cov2=cov
    #cov3=cov
    iter_res_1=[]
    iter_para_1=[]
    for iteration in range(1,31):
        #print(iteration)
        # E step
        x0=np.array(train_x2_1.iloc[:,:-1])
        p1=multivariate_normal.pdf(x0, mean=u1,cov=cov1)
        #print("p1:"+str(p1))
        p2=multivariate_normal.pdf(x0, mean=u2,cov=cov2)
        phi1=(p1*pi1/(p1*pi1+p2*pi2)) # the proba that each data fall in cluster 1 2019,0
        phi2=(p2*pi2/(p1*pi1+p2*pi2)) # the proba that each data fall in cluster 2
        #phi3=(p3*pi3/(p1*pi1+p2*pi2+p3*pi3)) # the proba that each data fall in cluster 3
        of=np.log(pi1*p1+pi2*p2).sum()
        #print(pi1,pi2,pi3)
        iter_para_1.append({"u1":u1,"u2":u2,"cov1":cov1,"cov2":cov2,"pi1":pi1,"pi2":pi2})
        iter_res_1.append(of)
        # M step
        n=phi1.shape[0]
        n1=phi1.sum()
        n2=phi2.sum()
        #n3=phi3.sum()
        # Update Pi
        pi1=n1/n
        pi2=n2/n
        #pi3=n3/n
        # Update uk  
        u1=(phi1.reshape(-1,+1)*x0).sum(axis=0)/n1
        u2=(phi2.reshape(-1,+1)*x0).sum(axis=0)/n2
        #u3=(phi3.reshape(-1,+1)*x0).sum(axis=0)/n3
        # update cov
        cov1=(phi1.reshape(-1,+1)*(x0-u1)).T.dot((x0-u1))/n1
        cov2=(phi2.reshape(-1,+1)*(x0-u2)).T.dot((x0-u2))/n2
        #cov3=(phi3.reshape(-1,+1)*(x0-u3)).T.dot((x0-u3))/n3   
        
        #iter_para.append((u1,u2,u3,cov1,cov2,cov3,p1,p2,p3))
    run_res_1.append(iter_res_1)
    run_para_1.append(iter_para_1)


# In[47]:


# make a prediction using 2GMM 

max_run_0=np.argmax(np.array(run_res_0).T[:][-1])
max_run_1=np.argmax(np.array(run_res_1).T[:][-1])
x0=test_x2
p1=multivariate_normal.pdf(x0, mean=run_para_0[max_run_0][-1]['u1'],cov=run_para_0[max_run_0][-1]['cov1'])
p2=multivariate_normal.pdf(x0, mean=run_para_0[max_run_0][-1]['u2'],cov=run_para_0[max_run_0][-1]['cov2'])
of_0=np.log(pi1*p1+pi2*p2)+np.log(prior_0)
# make a prediction 
x0=test_x2
p1=multivariate_normal.pdf(x0, mean=run_para_1[max_run_1][-1]['u1'],cov=run_para_1[max_run_1][-1]['cov1'])
p2=multivariate_normal.pdf(x0, mean=run_para_1[max_run_1][-1]['u2'],cov=run_para_1[max_run_1][-1]['cov2'])
of_1=np.log(pi1*p1+pi2*p2)+np.log(prior_1)

prediction=of_1>of_0


# ##### Confusion Matrix of 2 GMM prediction 

# In[48]:


from sklearn.metrics import confusion_matrix
matrix2=(confusion_matrix(test_y2, prediction))
accuracy2=(matrix2[0][0]+matrix2[1][1])/matrix2.sum()


# ### 4GMM MODEL 

# In[23]:


from scipy import stats
from scipy.stats import multivariate_normal
run_res_0=[]
run_para_0=[]
for run in range(1,11):
    #print(run)
    c=4
    pi1=1.0/4
    pi2=1.0/4
    pi3=1.0/4
    pi4=1.0/4
    mean=np.array(train_x2_0.iloc[:,:-1].mean())
    cov=np.cov(train_x2_0.iloc[:,:-1].T)
    u=np.random.multivariate_normal(mean,cov,4)
    u1=u[0]
    u2=u[1]
    u3=u[2]
    u4=u[3]
    cov1=cov
    cov2=cov
    cov3=cov
    cov4=cov
    iter_res_0=[]
    iter_para_0=[]
    for iteration in range(1,31):
        #print(iteration)
        # E step
        x0=np.array(train_x2_0.iloc[:,:-1])
        p1=multivariate_normal.pdf(x0, mean=u1,cov=cov1,allow_singular=True)
        p2=multivariate_normal.pdf(x0, mean=u2,cov=cov2,allow_singular=True)
        p3=multivariate_normal.pdf(x0, mean=u3,cov=cov3,allow_singular=True)
        p4=multivariate_normal.pdf(x0, mean=u4,cov=cov4,allow_singular=True)
        phi1=(p1*pi1/(p1*pi1+p2*pi2+p3*pi3+p4*pi4)) # the proba that each data fall in cluster 1 2019,0
        phi2=(p2*pi2/(p1*pi1+p2*pi2+p3*pi3+p4*pi4)) # the proba that each data fall in cluster 2
        phi3=(p3*pi3/(p1*pi1+p2*pi2+p3*pi3+p4*pi4)) # the proba that each data fall in cluster 3
        phi4=(p4*pi4/(p1*pi1+p2*pi2+p3*pi3+p4*pi4))
        of=np.log(pi1*p1+pi2*p2+pi3*p3+p4*pi4).sum()
        iter_para_0.append({"u1":u1,"u2":u2,"u3":u3,"u4":u4,"cov1":cov1,"cov2":cov2,"cov3":cov3,"cov4":cov4,"pi1":pi1,"pi2":pi2,"pi3":pi3,"pi4":pi4})
        iter_res_0.append(of)
        # M step
        n=phi1.shape[0]
        n1=phi1.sum()
        n2=phi2.sum()
        n3=phi3.sum()
        n4=phi4.sum()
        # Update Pi
        pi1=n1/n
        pi2=n2/n
        pi3=n3/n
        pi4=n4/n
        # Update uk  
        u1=(phi1.reshape(-1,+1)*x0).sum(axis=0)/n1
        u2=(phi2.reshape(-1,+1)*x0).sum(axis=0)/n2
        u3=(phi3.reshape(-1,+1)*x0).sum(axis=0)/n3
        u4=(phi4.reshape(-1,+1)*x0).sum(axis=0)/n4
        # update cov
        cov1=(phi1.reshape(-1,+1)*(x0-u1)).T.dot((x0-u1))/n1
        cov2=(phi2.reshape(-1,+1)*(x0-u2)).T.dot((x0-u2))/n2
        cov3=(phi3.reshape(-1,+1)*(x0-u3)).T.dot((x0-u3))/n3   
        cov4=(phi4.reshape(-1,+1)*(x0-u4)).T.dot((x0-u4))/n4 
        #iter_para.append((u1,u2,u3,cov1,cov2,cov3,p1,p2,p3))
    run_res_0.append(iter_res_0)
    run_para_0.append(iter_para_0)

#label 1
run_res_1=[]
run_para_1=[]
for run in range(1,11):
    #print(run)
    c=4
    pi1=1.0/4
    pi2=1.0/4
    pi3=1.0/4
    pi4=1.0/4
    mean=np.array(train_x2_1.iloc[:,:-1].mean())
    cov=np.cov(train_x2_1.iloc[:,:-1].T)
    u=np.random.multivariate_normal(mean,cov,4)
    u1=u[0]
    u2=u[1]
    u3=u[2]
    u4=u[3]
    cov1=cov
    cov2=cov
    cov3=cov
    cov4=cov
    iter_res_1=[]
    iter_para_1=[]
    for iteration in range(1,31):
        #print(iteration)
        # E step
        x0=np.array(train_x2_1.iloc[:,:-1])
        p1=multivariate_normal.pdf(x0, mean=u1,cov=cov1,allow_singular=True)
        #print("p1:"+str(p1))
        p2=multivariate_normal.pdf(x0, mean=u2,cov=cov2,allow_singular=True)
        #print("p2:"+str(p2))
        p3=multivariate_normal.pdf(x0, mean=u3,cov=cov3,allow_singular=True)
        p4=multivariate_normal.pdf(x0, mean=u4,cov=cov4,allow_singular=True)
        phi1=(p1*pi1/(p1*pi1+p2*pi2+p3*pi3+p4*pi4)) # the proba that each data fall in cluster 1 2019,0
        phi2=(p2*pi2/(p1*pi1+p2*pi2+p3*pi3+p4*pi4)) # the proba that each data fall in cluster 2
        phi3=(p3*pi3/(p1*pi1+p2*pi2+p3*pi3+p4*pi4)) # the proba that each data fall in cluster 3
        phi4=(p4*pi4/(p1*pi1+p2*pi2+p3*pi3+p4*pi4))
        of=np.log(pi1*p1+pi2*p2+pi3*p3+p4*pi4).sum()
        #print(pi1,pi2,pi3)
        iter_para_1.append({"u1":u1,"u2":u2,"u3":u3,"u4":u4,"cov1":cov1,"cov2":cov2,"cov3":cov3,"cov4":cov4,"pi1":pi1,"pi2":pi2,"pi3":pi3,"pi4":pi4})
        iter_res_1.append(of)
        # M step
        n=phi1.shape[0]
        n1=phi1.sum()
        n2=phi2.sum()
        n3=phi3.sum()
        n4=phi4.sum()
        # Update Pi
        pi1=n1/n
        pi2=n2/n
        pi3=n3/n
        pi4=n4/n
        # Update uk  
        u1=(phi1.reshape(-1,+1)*x0).sum(axis=0)/n1
        u2=(phi2.reshape(-1,+1)*x0).sum(axis=0)/n2
        u3=(phi3.reshape(-1,+1)*x0).sum(axis=0)/n3
        u4=(phi4.reshape(-1,+1)*x0).sum(axis=0)/n4
        # update cov
        cov1=(phi1.reshape(-1,+1)*(x0-u1)).T.dot((x0-u1))/n1
        cov2=(phi2.reshape(-1,+1)*(x0-u2)).T.dot((x0-u2))/n2
        cov3=(phi3.reshape(-1,+1)*(x0-u3)).T.dot((x0-u3))/n3   
        cov4=(phi4.reshape(-1,+1)*(x0-u4)).T.dot((x0-u4))/n4 
        #iter_para.append((u1,u2,u3,cov1,cov2,cov3,p1,p2,p3))
    run_res_1.append(iter_res_1)
    run_para_1.append(iter_para_1)

# make a prediction using 4GMM 
#proba0=run_para_0[max_run_0][-1]
max_run_0=np.argmax(np.array(run_res_0).T[:][-1])
max_run_1=np.argmax(np.array(run_res_1).T[:][-1])


# In[24]:


#proba0=run_para_0[max_run_0][-1]
x0=test_x2
p1=multivariate_normal.pdf(x0, mean=run_para_0[max_run_0][-1]['u1'],cov=run_para_0[max_run_0][-1]['cov1'])
p2=multivariate_normal.pdf(x0, mean=run_para_0[max_run_0][-1]['u2'],cov=run_para_0[max_run_0][-1]['cov2'])
p3=multivariate_normal.pdf(x0, mean=run_para_0[max_run_0][-1]['u3'],cov=run_para_0[max_run_0][-1]['cov3'])
p4=multivariate_normal.pdf(x0, mean=run_para_0[max_run_0][-1]['u4'],cov=run_para_0[max_run_0][-1]['cov4'])
of_0=np.log(pi1*p1+pi2*p2+pi3*p3+pi4*p4)+np.log(prior_0)
# make a prediction 
#proba0=run_para_0[max_run_0][-1]
x0=test_x2
p1=multivariate_normal.pdf(x0, mean=run_para_1[max_run_1][-1]['u1'],cov=run_para_1[max_run_1][-1]['cov1'])
p2=multivariate_normal.pdf(x0, mean=run_para_1[max_run_1][-1]['u2'],cov=run_para_1[max_run_1][-1]['cov2'])
p3=multivariate_normal.pdf(x0, mean=run_para_1[max_run_1][-1]['u3'],cov=run_para_1[max_run_1][-1]['cov3'])
p4=multivariate_normal.pdf(x0, mean=run_para_1[max_run_1][-1]['u4'],cov=run_para_1[max_run_1][-1]['cov4'])
of_1=np.log(pi1*p1+pi2*p2+pi3*p3+pi4*p4)+np.log(prior_1)
prediction=of_1>of_0


# ##### Confusion Matrix of 4 GMM prediction 

# In[25]:


from sklearn.metrics import confusion_matrix
matrix4=(confusion_matrix(test_y2, prediction))
accuracy4=(matrix4[0][0]+matrix4[1][1])/matrix4.sum()
print(matrix4)
print("Accuracy is {}".format(accuracy4))


# ### Confusion matrix for GMM 

# In[44]:


print("1GMM")
print(matrix4)
print("Accuracy is {}".format(accuracy1))
print("2GMM")
print(matrix2)
print("Accuracy is {}".format(accuracy2))
print("3GMM")
print(matrix3)
print("Accuracy is {}".format(accuracy3))
print("4GMM")
print(matrix4)
print("Accuracy is {}".format(accuracy4))


# ## Problem 3 (Matrix factorization)

# ### a)  
# Run your code 10 times. For each run, initialize your ui and vj vectors as N(0,I) random vectors. On a single plot, show the the log joint likelihood for iterations 2 to 100 for each run. In a table, show in each row the final value of the training objective function next to the RMSE on the testing set. Sort these rows according to decreasing value of the objective function.

# In[26]:


import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#read in data 
train_x3=pd.read_csv('/Users/zhejindong/Desktop/hw3-data/Prob3_ratings.csv',header=None)
test_x3=pd.read_csv('/Users/zhejindong/Desktop/hw3-data/Prob3_ratings_test.csv',header=None)
print(test_x3.shape)
print(train_x3.shape)


# In[27]:


# Unique users
print("Unique users:")
print(len(set(train_x3[0])))
# Unique movies
print("Unique movies:")
print(len(set(train_x3[1])))


# In[28]:


# max users
print("Unique users:")
print(max(set(train_x3[0])))
print(max(set(test_x3[0])))
# Unique movies
print("Unique movies:")
print(max(set(train_x3[1])))
print(max(set(test_x3[1])))


# In[29]:


row=943
col=1682
M=np.zeros((row+1,col+1))


# In[30]:


# Build a matrix for movie rating 
for i in train_x3.values:
    M[int(i[0])][int(i[1])]=i[2]


# In[31]:


M.shape


# In[32]:


# initialize the vectors for user and movie 
mean=np.zeros(10)
cov=np.identity(10, dtype = float) 


# In[33]:


# UPDATE U and V
outer=[]
outer_para=[]
for run in range(10):
    U=np.random.multivariate_normal(mean,cov,M.shape[0])
    V=np.random.multivariate_normal(mean,cov,M.shape[1])
    inner=[]
    for iteration in range(100):
        for i in range(M.shape[0]):
            U[i]=M[i][M[i]!=0].reshape(+1,-1).dot(V[M[i]!=0]).dot(inv(V[M[i]!=0].T.dot(V[M[i]!=0])+0.25*np.identity(10)))
        for j in range(M.shape[1]):
            V[j]=M[M[:,j]!=0,j].reshape(+1,-1).dot(U[M[:,j]!=0]).dot(inv(U[M[:,j]!=0].T.dot(U[M[:,j]!=0])+0.25*np.identity(10)))
        obj=np.log(norm.pdf(M,loc=U.dot(V.T),scale=0.5)[(M!=0)]).sum()+np.log(multivariate_normal.pdf(U,mean,cov)).sum()+np.log(multivariate_normal.pdf(V,mean,cov)).sum()
        inner.append(obj)
    outer_para.append({'U':U,'V':V})
    outer.append(inner)  
    


# In[34]:


import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(2,100),np.array(outer).T[2:])
plt.title("Objective_MF")


# In[35]:


# Prediction
from sklearn.metrics import mean_squared_error
rmse=[]
for run in range(10):
    prediction=np.diagonal(outer_para[run]['U'][test_x3[0]].dot(outer_para[run]['V'][test_x3[1]].T))
    rmse.append(mean_squared_error(test_x3[2],prediction))


# In[36]:


res=[]
for run in range(10):
    res.append((run+1,outer[run][-1],rmse[run]))

sorted_res=sorted(res,key=lambda tup: tup[1],reverse=True)


# In[37]:


print("run       objective_function           rmse")
print("---------------------------------------------------")
for i in sorted_res:
    print(str(i[0])+"        "+str(i[1])+"      "+str(i[2]))


# ### Comment:
# 
# from the table above, we can see as objective value decreases, the rmse begins decrease and then increases. We can interprete this process as from overfitting to underfitting. In the middle, we model finds a balance where rmse is smallest. 

# ### b)
# 
# For the run with the highest objective value, pick the movies “Star Wars” “My Fair Lady” and “Goodfellas” and for each movie find the 10 closest movies according to Euclidean distance usingtheir respective locations vj. List the query movie, the ten nearest movies and their distances.  A mapping from index to movie is provided with the data.

# In[38]:


movie=[]
file1 = open("/Users/zhejindong/Desktop/hw3-data/Prob3_movies.txt") 
for i in file1.readlines():
    movie.append(i.strip('\n'))


# In[39]:


import re 
r = re.compile("Star Wars*")
newlist = list(filter(r.match, movie)) # Read Note
movie_id=movie.index(newlist[0])+1
V_id=outer_para[9]['V'][movie_id]
nn10=np.argsort(dist(outer_para[9]['V'], V_id, ax=1))[1:11]
print("closest movie as \"Star War\"\n")

df={}
for i in nn10:
    #print(movie[i-1])
    df[movie[i-1]]=[dist(outer_para[9]['V'], V_id, ax=1)[i]]
    
pd.DataFrame.from_dict(data=df)


# In[40]:


r = re.compile("My Fair Lady*")
newlist = list(filter(r.match, movie)) # Read Note
movie_id=movie.index(newlist[0])+1
V_id=outer_para[9]['V'][movie_id]
nn10=np.argsort(dist(outer_para[9]['V'], V_id, ax=1))[1:11]
print("closest movie as \"My Fair Lady\"\n")

df={}
for i in nn10:
    #print(movie[i-1])
    df[movie[i-1]]=[dist(outer_para[9]['V'], V_id, ax=1)[i]]
    
pd.DataFrame.from_dict(data=df)


# In[41]:


r = re.compile("GoodFellas*")
newlist = list(filter(r.match, movie)) # Read Note
movie_id=movie.index(newlist[0])+1
V_id=outer_para[9]['V'][movie_id]
nn10=np.argsort(dist(outer_para[9]['V'], V_id, ax=1))[1:11]
print("closest movie as \"GoodFellas\"\n")

df={}
for i in nn10:
    #print(movie[i-1])
    df[movie[i-1]]=[dist(outer_para[9]['V'], V_id, ax=1)[i]]
    
pd.DataFrame.from_dict(data=df)

