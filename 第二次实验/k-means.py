from scipy.cluster.vq import kmeans, vq
from numpy import array, reshape, zeros
import cv2
import numpy as np

vqclst = [2, 10, 64]

#读入图片数据
data = cv2.imread('/Users/yangboqing/Downloads/timg.jpg').astype(float)
(height, width, channel) = data.shape

data = reshape(data, (height * width, channel))

for k in vqclst:
    print('Generating vq-%d...' % k)
    (centroids, distor) = kmeans(data, k)     #输出有两个,第一个是聚类中心(centroids),第二个是损失distortion,即聚类后各数据点到其聚类中心的距离的加和
    (code, distor) = vq(data, centroids)      #根据聚类中心将所有数据进行分类。输出同样有两个:第一个是各个数据属于哪一类的label,第二个和kmeans的第二个输出是一样的,都是distortion
    #print('distor: %.6f' % distor.sum())
    im_vq = centroids[code, :]
    img = reshape(im_vq, (height, width, channel))
    #产生进行压缩后的图片
    cv2.imwrite('result-%d.jpg' % k, img)
    #展示进行压缩后的图片
    img = img.astype(np.uint8)           #float的矩阵并不是归一化后的矩阵并不能保证元素范围一定就在0-1之间，所以要进行强制类型转换
    cv2.imshow('im%d' % k,img)
    cv2.waitKey(0)
    print(k,"finish")