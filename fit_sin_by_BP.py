"""
使用bp神经网络对y = sin(x),x属于(-pi,pi),进行拟合
使用批处理方法，每一轮更新一次参数
"""
import numpy as np
import BPnet_sin as BPnet
import matplotlib.pyplot as plt
import torch
import os
import time

def saveModel(lastLoss,net,path):
    """只在loss下降时保存模型"""
    if net.testLoss < lastLoss:
        checkpoint = {'testLoss': net.testLoss,
              'weights':net.w,
              'biases': net.b}
        torch.save(checkpoint, path)
        print("save better model")

def loadModel(path,net):
    lastLoss = np.inf
    if os.path.exists(path):
        checkpoint = torch.load(path)
        lastLoss = checkpoint['testLoss']
        net.loadPara(checkpoint['weights'],checkpoint['biases'])
    return lastLoss


def visualizeResult(trainX,trainY,testX,testY,testResult,trainLoss,testLoss):
    plt.figure("fitting result",figsize=(15,5))
    plt.subplot(131)
    plt.plot(trainX,trainY,'*',label='sample')
    plt.plot(testX,testResult,'b-',label='fitting result',linewidth=1) 
    plt.plot(testX,testY,'g--',label='standard sin',linewidth=1) 
    plt.legend()
    plt.subplot(132)
    plt.plot(trainLoss,label='train loss')
    plt.text(len(trainLoss)*0.8,trainLoss[-1],'%e'%trainLoss[-1])
    plt.legend()
    plt.subplot(133)
    plt.plot(testLoss,label='test loss')
    plt.text(len(testLoss)*0.8,testLoss[-1],'%e'%testLoss[-1])
    print('%e'%testLoss[-1])
    plt.legend()
    plt.show()

def getDataSet(trainNum,testRadio):
    #生成训练样本集
    trainX = np.linspace(-np.pi,np.pi,trainNum)
    trainY = np.sin(trainX)
    #生成测试样本集
    testX = np.linspace(-np.pi,np.pi,trainNum*testRadio).reshape(1,trainNum*testRadio)
    testY = np.sin(testX)
    data = np.append(testX,trainX)
    label = np.append(testY,trainY)
    data = data.reshape(1,len(data))
    label = label.reshape(1,len(label))
    return data,label,trainX,trainY,testX,testY

##参数
TRAINN = 20 #训练样本数
TESTRADIO = 5 #测试集：20*5 = 100个
EPOCH = 10000
R = 0.09
L = 0
NET_STRUCTURE = (1,30,30,1)
SAVEPATH = "./lab1/fitsin_parameter.pkl"

sigmoid = lambda x: 1/(1+np.exp(-x))
d_sigmoid = lambda x: sigmoid(x)*(1-sigmoid(x))
output_Active = lambda x:x #拟合，输出层的激活函数是恒等函数I
lossFun = lambda d,o:(((d-o).dot((d-o).T))*0.5)[0][0]
relu = lambda x:(np.abs(x) + x) / 2.0

def d_relu(x):
    y = np.zeros(x.shape)
    y[x > 0] = 1
    return y

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def d_tanh(x):
    return 1-(tanh(x))**2

if __name__=='__main__':
    data,label,trainX,trainY,testX,testY = getDataSet(TRAINN,TESTRADIO)
    BP = BPnet.BPnet(R,L,NET_STRUCTURE,[sigmoid,output_Active],d_sigmoid)
    lastLoss = loadModel(SAVEPATH,BP)
    startT = time.time()
    trainLoss,testLoss = BP.train(data,label,EPOCH,lossFun,testRadio=TESTRADIO)
    endT = time.time()
    print('训练用时：',endT-startT)
    testResult = BP.test(testX)
    visualizeResult(trainX,trainY,testX[0],testY[0],testResult[0],trainLoss,testLoss)
    saveModel(lastLoss,BP,SAVEPATH)
    input()