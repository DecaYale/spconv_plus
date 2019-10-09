#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '/home/yxu/MyFiles/Project/Github/spconv_1.0/')
sys.path.insert(0, '/home/yxu/MyFiles/Project/Github/spconv_1.0/spconv')
import numpy as np 
import torch
# import spconv 


# In[2]:


import spconv

# from spconv.conv import SparseConcat


# In[3]:


sparse_shape = [10, 20,30]
coors1_np=np.array([0,0,0,0,
                   0,5,5,5
                  ]).reshape(-1,4)
coors2_np=np.array([0,0,0,0,
                   0,5,6,6
                  ]).reshape(-1,4)
voxel_features1_np = np.array([0,0,
                             5,5]).reshape(-1,2)
voxel_features2_np = np.array([0,0,
                             6,6]).reshape(-1,2)
batch_size=1
device='cuda'


# In[4]:


# sparse_shape = torch.from_numpy(sparse_shape_np).to(device=device)
coors1=torch.from_numpy(coors1_np).to(device=device)
coors2=torch.from_numpy(coors2_np).to(device=device)

voxel_features1 = torch.from_numpy(voxel_features1_np).to(device=device)
voxel_features2 = torch.from_numpy(voxel_features2_np).to(device=device)
batch_size=1


# In[5]:



coors1 = coors1.int()
coors2 = coors2.int()
tensor1= spconv.SparseConvTensor(voxel_features1, coors1, sparse_shape,
                                      batch_size)
tensor2= spconv.SparseConvTensor(voxel_features2, coors2, sparse_shape,
                                      batch_size)


# In[6]:

# import pdb 
# pdb.set_trace()
sp_cat = spconv.SparseConcat3d(indice_key='spcat0')


# In[1]:

import pdb 
pdb.set_trace()
sp_cat(tensor1, tensor2)


# In[ ]:



