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
        self.testAccuracy = 0 #本模型在测试集上的准确率（如果没有超过上一次保存的模型的准确率，就不保存）
        # self.randomInitialize()
        self.xavierInitialize()

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
        plt.ion() #开启interactive mode,实时画图
        plt.figure(figsize=(10,4))
        if testRadio:
            x,y,testSet,testLabel = self.splitTestSet(x,y,testRadio)#划分测试集合testSet和训练集x
            allTestAccuracy = [] #记录验证集上准确率
        trainAccuracy = []
        n = x.shape[1] #训练样本数,每列是一个样本
        batchSize = n if batchSize == 0 else batchSize
        epochLoss = []
        for i in range(epoch):
            print('----epoch %s----'%(i+1))
            batchLoss = [] #记录一轮中每个batch的误差 
            batchPtr = 0 #记录下一轮batch训练开始的位置
            while batchPtr < n:
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
            # 每轮计算损失
            epochMeanLoss = sum(batchLoss)/n
            print('----epoch%s loss: %s----'%(i+1,epochMeanLoss))
            epochLoss.append(epochMeanLoss)
            # 每轮计算测试集和训练集上的准确率
            self.getEpochAccuracy(x,y,trainAccuracy,testSet,testLabel,allTestAccuracy)
            self.visualizeAccuracy(epochLoss,trainAccuracy,allTestAccuracy)
        # plt.savefig("classification.png")
        return epochLoss

    def getEpochAccuracy(self,train,trainLabel,trainAcc,testSet,testLabel,testAcc):
        """随机梯度下降时每个epoch结束计算整体训练集和测试集的准确率"""
        trainOut = self.test(train)
        self.calBatchAccuracy(trainOut,trainLabel,trainAcc,testSet,testLabel,testAcc)

    def calBatchAccuracy(self,trainOut,trainLabel,trainAcc,testSet,testLabel,testAcc):
        acc = self.getAccuracy(trainOut,trainLabel)#trainset上的准确率
        trainAcc.append(acc)
        print('train set accuracy: %f %%'%(acc*100)) #准确率
        result = self.test(testSet)
        self.testAccuracy = self.getAccuracy(result,testLabel)#test set
        print('test set accuracy: %f %%'%(self.testAccuracy*100)) #准确率
        testAcc.append(self.testAccuracy)

    def splitTestSet(self,x,y,testRadio):
        n = x.shape[1]
        testSize = math.ceil(n*testRadio) #测试集合样本数目
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

    def visualizeAccuracy(self,loss,acc1,acc2):
        plt.subplot(131)
        plt.title('loss')
        plt.plot(loss,'-b',linewidth = 1)

        plt.subplot(132)
        plt.title('train accuracy')
        plt.plot(acc1,'-b',linewidth = 1)

        plt.subplot(133)
        plt.title('test accuracy')
        plt.plot(acc2,'-b',linewidth = 1)
        plt.pause(0.001)
        plt.ioff()  # 关闭画图的窗口
    
    def loadPara(self,w,b):
        self.w = w
        self.b = b