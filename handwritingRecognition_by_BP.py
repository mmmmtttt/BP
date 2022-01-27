import os
import cv2
import numpy as np
import BPnet_word as BPnet
import torch
import time

CATEGORY = 12 #分类数目

def loadImage():
    np.random.seed(0) #生成随机数种子，保证每次shuffle的顺序一样
    X = []#输入，每个元素是长度为28*28的ndarray
    Y = []#标签，每个元素是一个值
    for i in range(1,CATEGORY+1):
        n = len(os.listdir("./lab1/train/%s" % i)) #得到文件总数
        for j in range(1,n+1):
            img = cv2.imread("./lab1/train/%s/%s.bmp"%(i,j),0)#读取灰度图
            X.append(img.flatten()/255)#拼接成一维的向量，并且归一化
            Y.append(i-1)
    x = np.array(X) #list转换成ndarry,每一行是一个样本
    y = np.zeros([len(Y),CATEGORY])#(样本数，12)
    y[range(len(Y)),Y] = 1 #黄金标签对应元素为1
    #打乱数据集
    state = np.random.get_state()#保持数据和标签按照相同的规则进行shuffle
    np.random.shuffle(x)#shuffle是打乱行
    np.random.set_state(state)
    np.random.shuffle(y)
    return x.T,y.T #转换成每列一个样本

def saveModel(lastAcc,net,path):
    """只在准确率提升时保存模型"""
    if net.testAccuracy > lastAcc:
        checkpoint = {'testAccuracy': net.testAccuracy,
              'weights':net.w,
              'biases': net.b}
        torch.save(checkpoint, path)

def loadModel(path,net):
    lastAccuracy = 0
    if os.path.exists(path):
        checkpoint = torch.load(path)
        lastAccuracy = checkpoint['testAccuracy']
        net.loadPara(checkpoint['weights'],checkpoint['biases'])
    return lastAccuracy

def cross_entropy(d,o):
    """
    损失函数：交叉熵
    d是所有样本的黄金标签矩阵，（12，样本数）
    o是所有样本在最后一层经过softmax激活后的输出集合
    返回的是总和，而非平均
    """
    return -np.sum(np.log(o[d==1])) #使用布尔索引取得o中黄金标签位置的元素

def softmax(x):
    z = x - np.max(x) #优化：防止溢出
    return np.exp(z)/np.sum(np.exp(z),axis = 0)

def sigmoid(x):
    """
    优化：防止sigmoid在计算绝对值很小的负数的exp时溢出
    """
    mask = (x > 0)
    positive = np.zeros(x.shape)
    negative = np.zeros(x.shape)
    # 大于0
    positive = 1 / (1 + np.exp(-x, positive, where=mask))#储存到positive，对于mask为true的位置计算
    positive[~mask] = 0
    # 小于等于0
    expZ = np.exp(x,negative,where=~mask)
    negative = expZ / (1+expZ)
    negative[mask] = 0
    return positive + negative

d_sigmoid = lambda x: sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def d_tanh(x):
    return 1-(tanh(x))**2

relu = lambda x:(np.abs(x) + x) / 2.0

def d_relu(x):
    y = np.zeros(x.shape)
    y[x > 0] = 1
    return y

I = lambda x:x #拟合，输出层的激活函数是恒等函数I
MSE = lambda d,o:(((d-o).dot((d-o).T))*0.5)[0][0]

EPOCH = 100
R = 0.01
L = 0.01
IMG_SHAPE = 28*28
NET_STRUCTURE = (IMG_SHAPE,100,12)
BATCHSIZE = 1
TESTRADIO = 0.2 #验证集占训练集的比例
SAVEPATH = "./lab1/classification_parameter.pkl"

def test(path):
    #读取图片
    X = []#输入，每个元素是长度为28*28的ndarray
    Y = []#标签，每个元素是一个值
    for i in range(1,CATEGORY+1):
        n = len(os.listdir("%s/%s" % (path,i))) #得到文件总数
        for j in range(1,n+1):
            img = cv2.imread("%s/%s/%s.bmp"%(path,i,j),0)#读取灰度图
            X.append(img.flatten()/255)#拼接成一维的向量，并且归一化
            Y.append(i-1)
    x = np.array(X) #list转换成ndarry,每一行是一个样本
    y = np.zeros([len(Y),CATEGORY])#(样本数，12)
    y[range(len(Y)),Y] = 1 #黄金标签对应元素为1
    x = x.T
    y = y.T
    #测试
    BP = BPnet.BPnet(R,L,NET_STRUCTURE,[sigmoid,softmax],d_sigmoid)
    loadModel(SAVEPATH,BP)
    result = BP.test(x)
    print(result)
    accuracy = BP.getAccuracy(result,y)
    print('准确率：%s %%'% (accuracy*100))

def train():
    x,y = loadImage()
    BP = BPnet.BPnet(R,L,NET_STRUCTURE,[sigmoid,softmax],d_sigmoid)
    lastAccuracy = loadModel(SAVEPATH,BP)
    startT = time.time()
    BP.train(x,y,EPOCH,cross_entropy,BATCHSIZE,TESTRADIO)
    endT = time.time()
    print('训练用时：',endT-startT)
    saveModel(lastAccuracy,BP,SAVEPATH)

if __name__=='__main__':   
    mode = input('训练输入train,测试输入test：')
    if mode == 'train':
        print('train start!')
        train()
    else:
        print('test start!')
        path = input('输入测试图片集路径：')
        test(path)
