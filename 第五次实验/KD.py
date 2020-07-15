import numpy as np
import pandas as pd
from numpy import *
from time import *
begin_time = time()

class Kd_node:
    value = []  # 节点值
    deep = None  # 节点深度
    feature = None  # 划分标志
    left = None  # 左子树
    right = None  # 右子树
    parent = None  # 父节点

def Processing(x):
    kd_var = np.var(x, axis=0)  # 计算特征值
    max_feature_index = np.argmax(kd_var)  # 选出最大值的索引
    node = Kd_node()
    node = BuildKdTree(node, x, max_feature_index, 0) # 递归分割数组
    return node

def BuildKdTree(kdnode, x, index, deep):
    x = np.array(x)
    x = x[np.lexsort(x[:, ::-(index + 1)].T)]
    # 按照中间值进行节点赋值
    x_number = np.size(x, 0)  # 计算有多少行
    x_lie = np.size(x, 1)  # 计算有多少行
    x_midd = x_number // 2  # 计算中间的值
    kdnode = Kd_node()
    kdnode.value = x[x_midd, :]
    kdnode.deep = deep
    kdnode.feature = index
    # 数据划分
    x_left = x[0:x_midd, :]
    x_right = x[x_midd + 1:, :]
    if x_number == 1:  # 一个元素直接赋值即可
        return kdnode
    elif x_number == 2:
        kdnode.left = BuildKdTree(kdnode.left, x_left, (index + 1) % x_lie, deep + 1)
        kdnode.left.parent = kdnode
        return kdnode
    else:
        kdnode.left = BuildKdTree(kdnode.left, x_left, (index + 1) % x_lie, deep + 1)
        kdnode.left.parent = kdnode
        kdnode.right = BuildKdTree(kdnode.right, x_right, (index + 1) % x_lie, deep + 1)
        kdnode.right.parent = kdnode
        return kdnode

def search(node, x):
    global nearestPoint
    global nearestValue
    nearestPoint = None
    nearestValue = 0
    def travel(node, depth=0):
        global nearestPoint
        global nearestValue
        if node != None:
            n = len(x)
            axis = depth % n  #维度
            if x[axis] < node.value[axis]:
                travel(node.left, depth + 1)
            else:
                travel(node.right, depth + 1)
            distNodeAndX = dist(x, node.value)  # 递归完后向父节点方向回溯
            if (nearestPoint is None):
                nearestPoint = node.value
                nearestValue = distNodeAndX
            elif (nearestValue > distNodeAndX):
                nearestPoint = node.value
                nearestValue = distNodeAndX      #确定当前点，更新最近的点，最近的距离
            if (abs(x[axis] - node.value[axis]) <= nearestValue): #确定是否需要去子节点的区域去找
                if x[axis] < node.value[axis]:
                    travel(node.right, depth + 1)
                else:
                    travel(node.left, depth + 1)
    travel(node)
    return nearestPoint

def dist(x1, x2):
    return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5

# 读取数据并将数据转换为矩阵形式
def read_data(path):
    # 读数据
    data = pd.read_csv('/Applications/JuniorII/媒体数据管理/实验/第五次实验/'+path+'/real.txt', header=None, sep=' ')
    # 给数据集的行列重新命名（处理为平常矩阵形式）
    data.drop(columns=[0], axis=1, inplace=True)
    data.columns = np.linspace(0, data.shape[1] - 1, data.shape[1], dtype=int)
    data = np.array(data)
    x = data[:, 5:]
    return x
def show(path,input):
    data = read_data(path)
    node = Processing(data)
    print(search(node, input))
    end_time = time()
    run_time = end_time - begin_time
    print('该程序运行时间:', run_time)

if __name__ == '__main__':
    input=[100.25,32.1]
    print("CA中距离"+str(input)+"最近的点:")
    show('CA',input)
    print("--------------------------")
    print("BJ中距离"+str(input)+"最近的点:")
    show('BJ',input)
    print("--------------------------")