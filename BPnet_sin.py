import numpy as np
import matplotlib.pyplot as plt
import math
import torch #仅用于保存和加载模型参数

class BPnet(torch.nn.Module):

    def __init__(self,r,l,neuron,activeFun,d_activeFun):
        super(BPnet, self).__init__()
        self.r = r #学习率
        self.l = l #正则项前的系数，表示模型的平滑程度
        self.neuron = neuron #元组，表示网络各层神经元数量（包括输入和输出层）
        self.layer = len(neuron)-1 #w和b列表中元素的个数
        self.activeFun = activeFun
        self.d_activeFun =d_activeFun
        self.testLoss = 0 #本模型在测试集上的loss（如果没有小于上一次保存的，就不保存）
        self.randomInitialize()
        # self.xavierInitialize()

    def randomInitialize(self):
        """随机初始化权重和偏置,fit sin时表现更好"""
        self.w ,self.b= [],[]
        for i in range(self.layer):
            self.w.append(np.random.normal(0,1, size=(self.neuron[i+1],self.neuron[i]))) #用均值为0，标准差为1的正态分布来随机初始化
            self.b.append(np.zeros((self.neuron[i+1],1))) 
    
    def xavierInitialize(self):
        """xavier初始化，用于分类数字"""
        self.b = [np.random.randn(i, 1) for i in self.neuron[1:]] #去掉输入层,随机初始化b,标准正态分布
        self.w = []
        # self.b = []
        for i in range(0, self.layer):
            # self.b.append(np.zeros((self.neuron[i+1],1))) 
            ub = np.sqrt(6.0 / (self.neuron[i] + self.neuron[i+1])) #初始化的上下限
            lb = -ub
            self.w.append(np.random.uniform(lb, ub,(self.neuron[i+1],self.neuron[i])))#均匀分布

    def train(self,x,y,epoch,lossFun,batchSize=0,testRadio = 0):
        # plt.ion() #开启interactive mode,实时画图
        # plt.figure(figsize=(10,4))
        if testRadio:
            allTestLoss = [] #记录验证集的loss
            x,y,testSet,testLabel = self.splitTestSet(x,y,testRadio)#划分测试集合testSet和训练集x
        n = x.shape[1] #训练样本数,每列是一个样本
        batchSize = n if batchSize == 0 else batchSize
        epochLoss = []
        for i in range(epoch):
            batchLoss = [] #记录一轮中每个batch的误差 
            batchPtr = 0 #记录下一轮batch训练开始的位置
            while batchPtr < n:
                print('----epoch %s----'%(i+1))
                #划分batch
                data = x[:,batchPtr:min(batchPtr+batchSize,n)]
                label = y[:,batchPtr:min(batchPtr+batchSize,n)]
                batchPtr += batchSize
                z = [] #每一层神经元的输入值
                a = [] #每一层神经元的激活值
                self.forward(data,a,z)
                self.backward(label,a,z)
                #计算损失
                loss = lossFun(label,a[-1])
                batchLoss.append(loss)
                #在验证集上测试
                if testRadio:
                    self.calTestLoss(testSet,testLabel,lossFun,allTestLoss)
                    # self.visualizeLoss(epochLoss,allTestLoss)
            epochMeanLoss = sum(batchLoss)/n
            print('----epoch%s loss: %s----'%(i+1,epochMeanLoss))
            epochLoss.append(epochMeanLoss)
        return epochLoss,allTestLoss
    
    def calTestLoss(self,testSet,testLabel,lossFun,allTestLoss):
        result = self.test(testSet)
        self.testLoss = lossFun(result,testLabel)/testSet.shape[1]
        print('test set loss:%s'%(self.testLoss)) 
        allTestLoss.append(self.testLoss)

    def splitTestSet(self,x,y,testRadio):
        n = x.shape[1]
        #testSize是trainSize的testRadio倍
        testSize = math.ceil(n*(1-1/(testRadio+1))) #测试集合样本数目
        testSet = x[:,0:testSize]
        testLabel = y[:,0:testSize]
        x = x[:,testSize:n]
        y = y[:,testSize:n]
        return x,y,testSet,testLabel

    def forward(self,x,a,z):
        """前向传递"""
        #输入层：输出=输入
        z.append(x)
        a.append(x) 
        #对于每一隐层
        for k in range(self.layer-1): 
            z.append(self.w[k].dot(a[k])+self.b[k])
            a.append(self.activeFun[0](z[-1])) 
        #输出层：
        z.append(self.w[self.layer-1].dot(a[self.layer-1])+self.b[self.layer-1])
        a.append(self.activeFun[1](z[-1])) 

    def backward(self,label,a,z):
        """反向传播"""
        #计算损失函数对参数的导数
        dw = [] 
        dz = [] 
        db = []
        #输出层
        dz.append((a[-1]-label))
        #各个隐层
        for k in reversed(range(self.layer)): 
            #更新各层调整后的w和b
            dw.insert(0,dz[0].dot(a[k].T)/dz[0].shape[1])
            db.insert(0,np.sum(dz[0], axis=1, keepdims=True) /dz[0].shape[1])
            dz.insert(0,self.w[k].T.dot(dz[0])*self.d_activeFun(z[k]))
            self.w[k] -= self.r*dw[0]
            self.b[k] -= self.r*db[0]

    def test(self,x):
        """在验证集上计算输出"""
        a = x
        for k in range(self.layer-1): 
            z = self.w[k].dot(a)+self.b[k]
            a = self.activeFun[0](z)
        #输出层：
        z=self.w[self.layer-1].dot(a)+self.b[self.layer-1]
        a=self.activeFun[1](z) 
        return a
                
    def getAccuracy(self,output,label):
        """
        得到验证集上的准确率
        output是经过softmax算出的各类的概率，(12,样本数)
        """
        n = label.shape[1] #验证样本数
        o = np.argmax(output,axis=0) #argmax按列得到预测出的类别的索引
        l = np.where(label.T==1)[1] #得到黄金label的索引
        acc = np.sum(o==l)/n
        return acc

    def visualizeLoss(self,trainLoss,testLoss):
        plt.subplot(121)
        plt.title('train loss')
        plt.plot(trainLoss,'-b')
        plt.subplot(122)
        plt.title('test loss')
        plt.plot(testLoss,'-b')
        plt.pause(0.001)
        plt.ioff()  # 关闭画图的窗口
    
    def loadPara(self,w,b):
        self.w = w
        self.b = b