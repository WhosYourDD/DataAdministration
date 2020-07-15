
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas import DataFrame

#获取所有图片的名称
allpicture=[i for i in os.listdir('/Applications/JuniorII/媒体数据管理/实验/第四次实验/cifar-10/cifar-10/train/0')]
#自定义获取两张图片  相似特征点个数  的函数
def GetsimByName(name1,name2):
    imga = cv2.imread(name1)
    imgb = cv2.imread(name2)
    img1 = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(descriptors_1,descriptors_2)
    return len(matches)

sim=[]    #存储相似度
ming=[]   #存储图片名称
for name in allpicture[0:]:
    sim.append(GetsimByName('/Applications/JuniorII/a.jpg','/Applications/JuniorII/媒体数据管理/实验/第四次实验/cifar-10/cifar-10/train/0/'+name))
    ming.append('/Applications/JuniorII/媒体数据管理/实验/第四次实验/cifar-10/cifar-10/train/0/'+name)
df=DataFrame({'sim':sim,'name':ming})
large=df.nlargest(10, 'sim')
large
#相似特征点最多的前10张图片

#展示这前10张图片
plt.figure(figsize=(1,1))
plt.imshow(cv2.imread('/Applications/JuniorII/a.jpg'))
plt.show()
plt.figure(figsize=(10,10))
for i in range(0,10):
    img = cv2.imread(list(large['name'])[i])
    plt.subplot(1,10,i+1)
    plt.title(allpicture[i])
    plt.imshow(img)
plt.show()





