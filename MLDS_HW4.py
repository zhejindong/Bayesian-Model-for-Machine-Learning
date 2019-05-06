
# coding: utf-8

# ## Problem 1 (Markov chains)
# 
# You will rank 767 college football teams based on the scores of every game in the 2018 season. The data provided in CFB2018 scores.csv contains the result of one game on each line inthe format：
# 
# Team A index, Team A points, Team B index, Team B points
# 
# If Team A has more points than Team B, then Team A wins, and vice versa. The index of a team refers to the row of “TeamNames.txt” where that team’s name can be found. Construct a 767×767 random walk matrix M on the college football teams. First construct the unnor-malized matrix M' with values initialized to zeros. For one particular game, let i be the index of Team A and j the index of Team B. Then update M'
# 
# After processing all games, let M be the matrix formed by normalizing the rows of M' so they sum toone.  Letwtbe the 1×767 state vector at stept. Set w0 to the uniform distribution. Therefore,wt is the marginal distribution on each state after t steps given that the starting state is chosen uniformly atrandom.

# In[329]:


import pandas as pd 
import numpy as np
data=pd.read_csv("/Users/zhejindong/Desktop/hw4_data/CFB2018_scores.csv",header=None)


# In[295]:


data.shape


# In[296]:


data=np.array(data)


# In[297]:


import numpy as np
size=data.max()


# In[298]:


# Initialize M'
M=np.zeros((size+1,size+1))


# In[299]:


# update M'
for d in data:
    i=d[0]
    j=d[2]
    if d[1]>d[3]:
        # i wins 
        M[i][i]=M[i][i]+1+d[1]*1.0/(d[1]+d[3])
        M[j][j]=M[j][j]+d[3]*1.0/(d[1]+d[3])
        M[j][i]=M[j][i]+1+d[1]*1.0/(d[1]+d[3])
        M[i][j]=M[i][j]+d[3]*1.0/(d[1]+d[3])
    else:
        #j wins 
        M[i][i]=M[i][i]+d[1]*1.0/(d[1]+d[3])
        M[j][j]=M[j][j]+1+d[3]*1.0/(d[1]+d[3])
        M[i][j]=M[i][j]+1+d[3]*1.0/(d[1]+d[3])
        M[j][i]=M[j][i]+d[1]*1.0/(d[1]+d[3])


# In[300]:


M[1:,1:]


# In[301]:


# normalize the matrix:
M_1=(M[1:]/np.sum(M[1:],axis=1).reshape(-1,+1))
M=np.vstack((M[0],M_1))


# In[302]:


w=np.ones((1,size))/size


# ### a)  
# Use wt to rank the teams by sorting in decreasing value according to this vector.  List the top 25 team names (see accompanying file) and their corresponding values inwtfort= 10,100,1000,10000.

# In[303]:


name=pd.read_csv("/Users/zhejindong/Desktop/hw4_data/TeamNames.txt",header=None)


# In[304]:


def rank(t):
    w=np.ones((1,size))/size
    x=w.dot(np.linalg.matrix_power(M[1:,1:],t))
    top_25=np.argsort(x)[0][-25:]
    n=name.iloc[list(reversed(top_25)),0].values
    score=x[0][list(reversed(top_25))]
    return pd.DataFrame(score,index=n,columns=["rank={t}".format(t=t)])


# In[305]:


rank(10)


# In[306]:


rank(100)


# In[307]:


rank(1000)


# In[373]:


rank(10000)


# ### b) 
# We saw thatw∞is related to the first eigenvector ofMT. That is, we can findw∞by getting thefirst eigenvector and eigenvalue ofMTand post-processing:MTu1=λ1u1,  w∞=uT1/[∑ju1(j)]This is becauseuT1u1= 1by convention.  Also, we observe thatλ1= 1for this specific matrix.Plot‖wt−w∞‖1as a function oftfort= 1,...,10000.

# In[308]:


from numpy import linalg as LA


# In[309]:


w,v = LA.eig(M[1:,1:].T)
v = v.T


# In[310]:


rearrage=sorted(zip(w,v),key=lambda x:x[0],reverse=True)


# In[311]:


w_inf=rearrage[0][1]/(rearrage[0][1].sum())


# In[317]:


temp=[]
w=np.ones((1,size))/size
m=M[1:,1:]
for i in range(10000):
    x=w.dot(m)
    m=m.dot(M[1:,1:])
    score=x[0]
    temp.append(LA.norm(score-w_inf, 1))


# In[323]:


import matplotlib.pyplot as plt
plt.plot(temp)
plt.title("L2 distance with w_inf")
plt.show()


# ## Problem 2 (Nonnegative matrix factorization)
# 
# In this problem you will factorize an N×M matrix X into a rank-K approximation WH, where W is N×K, H is K×M and all values in the matrices are nonnegative. Each value in W and H can be initialized randomly to a positive number, e.g., from a Uniform(1,2) distribution. The data to be used for this problem consists of 8447 documents from The New York Times. (See belowfor how to process the data.)  
# 
# The vocabulary size is 3012 words. You will need to use this data to construct the matrix X, where Xij is the number of times word i appears in document j.  Therefore,X is 3012×8447 and most values in X will equal zero.

# In[2]:


corpus=pd.read_csv("/Users/zhejindong/Desktop/hw4_data/nyt_vocab.dat",header=None)


# In[3]:


f = open("/Users/zhejindong/Desktop/hw4_data/nyt_data.txt", "r")
text=f.readlines()


# In[4]:


x=np.zeros((len(corpus)+1,len(text)+1))


# In[5]:


for j,t in enumerate(text,1):
    temp=t.strip().split(',')
    for w_no,word in enumerate(temp):
        x[int(word.split(':')[0])][j]=int(word.split(':')[1])


# ## a)  
# Implement and run the NMF algorithm on this data using the divergence penalty. Set the rank to 25 and run for 100 iterations. This corresponds to learning 25 topics. Plot the objective as a function of iteration.

# In[173]:


# Initialize the matrix W and H
rank=25
W=np.random.rand(x[1:,1:].shape[0],rank)
H=np.random.rand(rank,x[1:,1:].shape[1])


# In[183]:


obj=[]
# update h and w 
for iteration in range(100):
        H=H*W.T.dot(x[1:,1:]/(W.dot(H)+1e-16))/W.sum(axis=0,keepdims=True).T
        W=W*(x[1:,1:]/(W.dot(H)+1e-16)).dot(H.T)/H.sum(axis=1,keepdims=True).T
        OB=(np.log(1/(W.dot(H)+1e-16))*x[1:,1:]+W.dot(H)).sum()
        obj.append(OB)


# In[324]:


import matplotlib.pyplot as plt
plt.plot(obj)
plt.title('objective function kl divergence')
plt.show()


# ## b)  
# After running the algorithm, normalize the columns of W so they sum to one. For each column of W, list the 10 words having the largest weight and show the weight. The ith row of W corresponds to the ith word in the “dictionary” provided with the data. Organize these lists in a 5×5table.

# In[228]:


# normalize the column of W
W1=W/(W.sum(axis=0,keepdims=True)+1e-16)


# In[346]:


p=[]
for i in range(25):
    index=np.argsort(W1[:,i])[-10:]
    index=list(reversed(index))
    p.append(pd.DataFrame(W1[index,i],index=corpus.iloc[index][0],columns=["column={t}".format(t=i)]))
    print(p[-1])

