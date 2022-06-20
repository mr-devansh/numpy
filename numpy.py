#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# ## DataTypes & Attributes

# In[2]:


#numpy has a main datatype that is ndarray
a1 = np.array([1,2,3])
a1


# In[3]:


type(a1)


# In[4]:


a2 = np.array([[1,2,3],
                [4,5,6]])
a2


# In[5]:


a3 = np.array([[[1,2,3],
               [4,5,6],
               [7,8,9]],
               [[10,11,12],
               [13,14,15],
               [16,17,18]]])
a3


# In[6]:


a1.shape


# In[7]:


a2.shape


# In[8]:


a3.shape


# In[9]:


a1.ndim, a2.ndim, a3.ndim


# In[10]:


a1.dtype, a2.dtype, a3.dtype


# In[11]:


a1.size, a2.size, a3.size


# In[12]:


type(a1), type(a2), type(a3)


# In[13]:


# we can create dataframes from ndarray from numpy in pandas
import pandas as pd
df = pd.DataFrame(a2)
df


# ## CREATE NUMPY ARRAYS

# In[14]:


sample_array = np.array([1,2,3])
sample_array


# In[15]:


# arrays filled with 1s and shape given
ones = np.ones((2,3))
ones


# In[16]:


ones.dtype


# In[17]:


type(ones)


# In[18]:


zeros = np.zeros((2,3))
zeros


# In[19]:


range_array = np.arange(0,10,2)
range_array


# In[20]:


random_array = np.random.randint(0,10,size=(3,5))
random_array


# In[21]:


np.random.random((5,3))


# ## VIEWING MATRICES AND ARRAY

# In[22]:


np.unique(random_array)


# In[23]:


a1


# In[24]:


a2


# In[25]:


a3


# In[26]:


a1[:2]


# In[27]:


a2[:2]


# In[28]:


a3[:2, :2, :2]


# In[29]:


a4 = np.random.randint(10, size=(2,3,4,5))
a4


# In[30]:


a4[:,:,:,:3]


# ## MANIPULATING AND COMPARING ARRAYS

# ### ARITHEMATIC

# In[31]:


a1


# In[32]:


ones = np.ones(3)
ones


# In[33]:


a1+ones


# In[34]:


a1-ones


# In[35]:


a1*ones


# In[36]:


a2


# In[37]:


# multiplies a1 with both rows of a2
a1*a2


# In[38]:


a1/ones


# In[39]:


a2//a1


# In[40]:


a1%2


# In[41]:


np.exp(a1)


# In[42]:


np.log(a1)


# ### AGGREGATION

# In[43]:


listy = [1,2,3]
type(listy)


# In[44]:


sum(listy)


# In[45]:


np.sum(listy)


# In[46]:


massive = np.random.random(100000)
massive


# In[47]:


#use `sum()` with pandas datatypes and `np.sum()` with numPy datatypes
get_ipython().run_line_magic('timeit', 'sum(massive)')
get_ipython().run_line_magic('timeit', 'np.sum(massive)')


# In[48]:


np.mean(a2)


# In[49]:


np.max(a2)


# In[50]:


np.min(a2)


# In[51]:


# square root of variance
np.std(a2)


# In[52]:


# deviation of each number from mean
np.var(a2)


# In[53]:


high_var_array = np.array([1,100,200,300,400,500,600,700,800,900,1000])
low_var_array = np.array([1,10,20,30,40,50,60,70,80,90,100])


# In[54]:


np.var(high_var_array), np.var(low_var_array)


# In[55]:


np.std(high_var_array), np.std(low_var_array)


# ### RESHAPING AND TRANSPOSING

# In[56]:


a3


# In[57]:


a3.shape


# In[58]:


a2


# In[59]:


a2_reshape = a2.reshape(2,3,1)
a2_reshape


# In[60]:


a2_reshape*a3


# In[61]:


# transpose: swaps axis
a2.T


# In[62]:


a3


# In[63]:


a3.T


# In[64]:


np.random.seed(0)
mat1 = np.random.randint(10, size=(5,3))
mat2 = np.random.randint(10, size=(5,3))
mat1, mat2


# In[65]:


# elemwnt wise multiplication
mat1*mat2


# In[66]:


# dot product
np.dot(mat1.T,mat2)


# ### COMPARISON OPERATORS

# In[68]:


a1>a2


# In[70]:


# > < >= <= == != 


# ## SORTING ARRAYS

# In[71]:


random_array


# In[74]:


np.sort(random_array)


# In[75]:


np.argsort(random_array)


# In[77]:


np.argmax(random_array, axis=0)


# In[78]:


np.argmin(random_array, axis=1)


# ## IMAGES TO DATAFRAMES

# In[87]:


from IPython.display import Image
Image("image2.jpg")


# In[88]:


# turn into array
from matplotlib.image import imread
pic = imread("image2.jpg")
pic


# In[89]:


pic.size, pic.shape, pic.ndim


# In[95]:


pic[:1]

