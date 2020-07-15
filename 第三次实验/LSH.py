#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
import random
import hashlib
# 分别进行近邻搜索，查询点为数据集前1000点，查找前10个最近邻，统计搜索算法的性能（召回率，准确率，时间
# In[2]:
data = pd.read_csv("/Applications/Projects/ColorHistogram.asc", sep=" ", header=None, index_col=0)
data.head()
# In[3]:
dataSet = data.values
# In[4]:
dataSet.shape
# In[18]:
from copy import copy
from itertools import combinations
import numpy as np
from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances

class LSH:
    def __init__(self, data):
        self.data = data
        self.model = None
    def __generate_random_vectors(self, num_vector, dim):
        return np.random.randn(dim, num_vector)
    def train(self, num_vector, seed=None):
        dim = self.data.shape[1]
        if seed is not None:
            np.random.seed(seed)      
        #随机生成正态分布矩阵
        random_vectors = self.__generate_random_vectors(num_vector, dim)
        
        # 512 256 128 64 32 16 8 4 2 1 
        powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)
        table = {}
        #数据按正负分类
        bin_index_bits = (self.data.dot(random_vectors) >= 0)#68040*5
        print(bin_index_bits)
        bin_indices = bin_index_bits.dot(powers_of_two)  #68040个key值
        print(bin_indicies)
        # 将 68040个索引key值进行分类，相同的key值对应的id映射到一个桶中
        for data_index, bin_index in enumerate(bin_indices):  #enumerate()同时列出数据和数据下标
            if bin_index not in table:
                # 判断当前key值桶里是否存在，不存在就新建一个
                table[bin_index] = []
            #存在就加入数据
            table[bin_index].append(data_index)

        self.model = {'bin_indices': bin_indices, 'table': table,
                      'random_vectors': random_vectors, 'num_vector': num_vector}
        return self
    def __search_nearby_bins(self, query_bin_bits, table, search_radius=2, initial_candidates=set()):
        num_vector = self.model['num_vector']
        powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)
        candidate_set = copy(initial_candidates)
        for different_bits in combinations(range(num_vector), search_radius):
            alternate_bits = copy(query_bin_bits)
            for i in different_bits:
                alternate_bits[i] = 1 if alternate_bits[i] == 0 else 0
            # 2进制数据-->10进制
            nearby_bin = alternate_bits.dot(powers_of_two)
            # 得到该向量的所有临近桶key值，通过该key值找id，找到后加入同一个列表
            if nearby_bin in table:
                candidate_set.update(table[nearby_bin])
        return candidate_set   #返回的是（list列表）包含该点的近邻点的所有id索引值，从而缩小了查近邻点的查询范围

    def query(self, query_vec, k, max_search_radius):
                    #查询k近邻点
        if not self.model:
            print('Model not yet build. Exiting!')
            exit(-1)
        data = self.data
        table = self.model['table']
        random_vectors = self.model['random_vectors']
        initial_candidates=set()
        bin_index_bits = (query_vec.dot(random_vectors) >= 0).flatten()  #bin_index_bits 1*5的T和F
        candidate_set = set()
        # 在搜索半径内，查询该向量的所有索引key值
        for search_radius in range(max_search_radius + 1):
            candidate_set = self.__search_nearby_bins(bin_index_bits, table,
                                                      search_radius, initial_candidates=initial_candidates)
            initial_candidates = candidate_set
        # 拿到返回的列表，在该列表内暴力查找最近邻点，用datafram分别存储id和distance
        nearest_neighbors = DataFrame({'id': list(candidate_set)})
        candidates = data[np.array(list(candidate_set)), :]
        nearest_neighbors['distance'] = pairwise_distances(candidates, query_vec.reshape(1,-1), metric='cosine').flatten()
        return nearest_neighbors.nsmallest(k, 'distance')  #按照距离求出前k个最近邻点

# In[6]:
lsh_model = LSH(dataSet)
# In[19]:
num_of_random_vectors = 10
lsh_model.train(num_of_random_vectors)
# In[8]:
lsh_model.query(dataSet[1,:], 11, 2
# #暴力法求前1000个点的最近邻点

# In[9]:


#前1000个点在68040个点中暴力求距离，写入到real列表
real = [pairwise_distances(dataSet[:,:], dataSet[i,:].reshape(1,-1), metric='cosine').flatten() for i in range(1000)]
#对距离排序，将排序好的距离对应的id值写入real_index中
real_index = [np.argsort(real[i]) for i in range(1000)]


# In[10]:

import time
# 调参1，找随机变量维数  使得  正确率最高
history_norv = []
time1=[]
for i in range(3, 8):
    lsh_model.train(i)
    correct = 0
    t=time.time()
    for i in range(1000):
        lsh_res = lsh_model.query(dataSet[i,:], 11, 2)['id'].values[1:]
        correct += len(np.intersect1d(lsh_res, real_index[i][1:11]))
    t = time.time()-t
    history_norv.append(correct)
    time1.append(t)


# In[11]:


BestNum = history_norv.index(max(history_norv))+3
history_norv,BestNum


# In[12]:
time1
# In[13]:
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
bestnum=[j/10000 for j in history_norv]
plt.subplot(243)
plt.plot(range(3,8),bestnum,'b',label="accuracy")
plt.xlabel('num')
plt.ylabel('accuracy')
plt.ylim(0.95,1.05)

plt.subplot(244)
plt.plot(range(3,8),time1,'r',label="time")
plt.xlabel('num')
plt.ylabel('time/s')
plt.ylim(min(time1)*0.8, max(time1)*1.2,)
plt.legend()   #展示样例
plt.show()


# 综合考量，当随机变量个数选择5时，准确率较高，同时花费时间也比较短
# 也可以看出不使用LSH时，遍历一次只需要花费40s，暴力遍历需要60s以上

# In[14]:
num=5
# In[15]:
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# 调参2，寻找  最大搜索半径 和 花费时间的关系
lsh_model.train(num)
history_radius = []
for radius in range(0,5):
    t = time.time()
    correct = 0
    for i in range(1000):
        lsh_res = lsh_model.query(dataSet[i,:], 11, radius)['id'].values[1:]
        correct += len(np.intersect1d(lsh_res, real_index[i][1:11]))
    t = time.time()-t
    history_radius.append((t, correct))


# In[16]:
history_radius
# In[17]:
#绘图，数据展示
radius_time = [i[0] for i in history_radius]        #所花时间
radius_right = [i[1]/10000 for i in history_radius] #准确的个数

plt.subplot(241)
plt.plot([0,1,2,3,4], radius_time,'r',label="time")
plt.xlabel('radius')
plt.ylabel('time/s')
plt.ylim(min(radius_time)*0.8, max(radius_time)*1.2,)
plt.legend()   #展示样例

plt.subplot(242)
plt.plot(range(5),radius_right,'b',label="accuracy")
plt.xlabel('radius')
plt.ylabel('accuracy')
plt.ylim(min(radius_right)*0.8,1.1)
plt.legend()   #展示样例

plt.show()


# 可以看到，当搜索半径达到2时，准确率已经达到99%以上，所以搜索半径选择2比较合适
