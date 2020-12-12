import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
import math


#读取数据
df_plane = pd.read_csv('flight_303.csv')
df_port = pd.read_csv('InputData.csv')

#计算适应度，也就是距离分之一，这里用伪欧氏距离
def calcfit(gene):
    lamb = 0.02
    beta = 3
    sum = 0
    for item in gene:
        if item != -1:
            sum += 1
    z = math.exp(sum*lamb) * 10**beta
    return z

# 约束条件
def is_type_valid(plane, port):
    type1 = df_plane['type'][plane]
    type2 = df_port['type'][port]
    return type1==type2

def is_flight_valid(flight, port):
    arrive = df_plane['arrive_type'][flight] in df_port['arrive_type'][port]
    depart = df_plane['depart_type'][flight] in df_port['depart_type'][port]
    return arrive and depart

def is_time_valid(flight, port, table):
    valid = True
    conflict = table[port]
    flight_arrive_str = df_plane['arrive_date'][flight].strip()+df_plane['arrive_time'][flight].strip()
    flight_arrive = datetime.strptime(flight_arrive_str, '%d-%m-%y%H:%M')
    flight_depart_str = df_plane['depart_date'][flight].strip() + df_plane['depart_time'][flight].strip()
    flight_depart = datetime.strptime(flight_depart_str, '%d-%m-%y%H:%M') + timedelta(minutes=45)
    for item in conflict:
        item_arrive_str = df_plane['arrive_date'][item].strip()+df_plane['arrive_time'][item].strip()
        item_arrive = datetime.strptime(item_arrive_str, '%d-%m-%y%H:%M')
        item_depart_str = df_plane['depart_date'][item].strip()+df_plane['depart_time'][item].strip()
        item_depart = datetime.strptime(item_depart_str, '%d-%m-%y%H:%M') + timedelta(minutes=45)
        if (flight_arrive < item_arrive and flight_depart > item_arrive) or (flight_arrive > item_arrive and
            flight_arrive < item_depart):
            valid = False
    return valid

#每个个体的类，方便根据基因计算适应度
class Person:
    def __init__(self,gene):
        self.gene=gene
        self.fit=calcfit(gene)
class Group:
    def __init__(self):
        self.GroupSize=500  #种群规模
        self.GeneSize=303    #基因数量，也就是城市数量
        self.initGroup()
        self.upDate()
    #初始化种群，随机生成若干个体
    def initGroup(self):
        self.group=[]
        R = [[] for item in range(df_port.shape[0])]  # 冲突集合
        i=0
        while i < self.GroupSize:
            count = 0
            gene = []
            i += 1
            first = list(range(df_port.shape[0]))
            random.shuffle(first)
            # print('first', first)
            # 为每对航班分配登机口，若都不符合约束条件则赋-1
            while count < self.GeneSize:
                temp = -1
                for port in first:
                    # 判断该航班是否符合
                    constrain_type = is_type_valid(count, port)
                    constrain_arrive_depart = is_flight_valid(count, port)
                    constrain_time = is_time_valid(count, port, R)
                    # 如果都能满足
                    if constrain_type and constrain_arrive_depart and constrain_time:
                        temp = port
                        R[port].append(count)
                        break
                gene.append(temp)
                count += 1
            # print(df_plane['code'][0])
            print(gene)
            tmpPerson=Person(gene)
            self.group.append(tmpPerson)

    #获取种群中适应度最高的个体
    def getBest(self):
        bestFit=self.group[0].fit
        best=self.group[0]
        for person in self.group:
            if(person.fit>bestFit):
                bestFit=person.fit
                best=person
        return best
    #计算种群中所有个体的平均距离
    def getAvg(self):
        sum=0
        for p in self.group:
            sum+=1/p.fit
        return sum/len(self.group)
    #根据适应度，使用轮盘赌返回一个个体，用于遗传交叉
    def getOne(self):
        #section的简称，区间
        sec=[0]
        sumsec=0
        for person in self.group:
            sumsec+=person.fit
            sec.append(sumsec)
        p=random.random()*sumsec
        for i in range(len(sec)):
            if(p>sec[i] and p<sec[i+1]):
                #这里注意区间是比个体多一个0的
                return self.group[i]
    #更新种群相关信息
    def upDate(self):
        self.best=self.getBest()

#遗传算法的类，定义了遗传、交叉、变异等操作
class GA:
    def __init__(self):
        self.group=Group()
        self.pCross=0.35    #交叉率
        self.pChange=0.1    #变异率
        self.Gen=1  #代数

    def fix(self, gene):
        R = [[] for item in range(df_port.shape[0])]
        fixed_gene = []
        for index,item in enumerate(gene):
            if index == 0:
                R[item].append(index)
                fixed_gene.append(item)
                continue
            if item == -1:
                fixed_gene.append(item)
                continue
            constrain_type = is_type_valid(index, item)
            constrain_arrive_depart = is_flight_valid(index, item)
            constrain_time = is_time_valid(index, item, R)
            # 如果都能满足
            if constrain_type and constrain_arrive_depart and constrain_time:
                R[item].append(index)
                fixed_gene.append(item)
            else:
                # 修正当前位置
                temp = -1
                remain = list(set(df_port.index).difference(set(fixed_gene)))
                random.shuffle(remain)
                for port in remain:
                    # 判断该航班是否符合
                    constrain_type = is_type_valid(index, port)
                    constrain_arrive_depart = is_flight_valid(index, port)
                    constrain_time = is_time_valid(index, port, R)
                    # 如果都能满足
                    if constrain_type and constrain_arrive_depart and constrain_time:
                        temp = port
                        R[port].append(index)
                        break
                fixed_gene.append(temp)
        print('fixed_gene', fixed_gene)
        return fixed_gene

    #变异操作
    def change(self,gene):
        #把列表随机的一段取出然后再随机插入某个位置
        #length是取出基因的长度，postake是取出的位置，posins是插入的位置
        geneLenght=len(gene)
        index1 = random.randint(0, geneLenght - 1)
        index2 = random.randint(0, geneLenght - 1)
        newGene = gene[:]       # 产生一个新的基因序列，以免变异的时候影响父种群
        newGene[index1], newGene[index2] = newGene[index2], newGene[index1]
        return newGene

    #交叉操作
    def cross(self,p1,p2):
        geneLenght=len(p1.gene)
        index1 = random.randint(0, geneLenght - 1)
        index2 = random.randint(index1, geneLenght - 1)
        newGene = p1.gene.copy()
        newGene[index1:index2] = p2.gene[index1:index2].copy()
        '''
        tempGene = p2.gene[index1:index2]   # 交叉的基因片段
        newGene = []
        p1len = 0
        for g in p1.gene:
              if p1len == index1:
                    newGene.extend(tempGene)     # 插入基因片段
                    p1len += 1
              if g not in tempGene:
                    newGene.append(g)
                    p1len += 1
        '''
        # 变异为可行解
        print('p1:', p1.gene)
        print('p2:', p2.gene)
        print('cross: ', newGene)
        return self.fix(newGene)
        # return newGene


    #获取下一代
    def nextGen(self):
        self.Gen+=1
        #nextGen代表下一代的所有基因
        nextGen=[]
        #将最优秀的基因直接传递给下一代
        nextGen.append(self.group.getBest().gene[:])
        print('最优秀基因:', self.group.getBest().gene[:])
        while len(nextGen)<self.group.GroupSize:
            # pChange=random.random()
            pCross=random.random()
            p1=self.group.getOne()
            # 交叉
            if pCross<self.pCross:
                p2=self.group.getOne()
                newGene=self.cross(p1,p2)
                print('修正')
                print('newGene: ', newGene)
            else:
                newGene=p1.gene[:]
            # 变异
            # if pChange<self.pChange:
                # newGene=self.change(newGene)
            print('iter:', len(newGene))
            nextGen.append(newGene)
        print('nextGen: ', len(nextGen))
        self.group.group=[]
        for gene in nextGen:
            print('gene:', gene)
            self.group.group.append(Person(gene))
            self.group.upDate()

    #打印当前种群的最优个体信息
    def showBest(self):
        print("第{}代\t当前最优{}\t当前平均{}\t".format(self.Gen,1/self.group.getBest().fit,self.group.getAvg()))

    #n代表代数，遗传算法的入口
    def run(self,n):
        Gen=[]  #代数
        dist=[] #每一代的最优距离
        avgDist=[]  #每一代的平均距离
        #上面三个列表是为了画图
        i=1
        while(i<n):
            self.nextGen()
            self.showBest()
            i+=1
            Gen.append(i)
            dist.append(1/self.group.getBest().fit)
            avgDist.append(self.group.getAvg())
        #绘制进化曲线
        plt.plot(Gen,dist,'-r')
        plt.plot(Gen,avgDist,'-b')
        plt.show()

ga=GA()
ga.run(100)
print("进行3000代后最优解：",1/ga.group.getBest().fit)