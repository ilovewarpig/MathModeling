import random
import math
import matplotlib.pyplot as plt

def calcfit(x):
    x1, x2 = x
    y = 21.5 + x1*math.sin(4*math.pi*x1) + x2 * math.sin(20*math.pi*x2)
    return y

#鸟个体的类，实现鸟位置的移动
class Bird:
    def __init__(self,x):
        self.x=x
        self.vmax=1.2
        self.v = [-self.vmax + 2 * self.vmax * random.random() for vi in range(len(x))]
        #初始化时自己曾遇到得最优位置就是初始化的位置
        self.bestAddr=x
        #初始状态没有速度
        self.fit=calcfit(self.x)
        self.bestFit=self.fit

    #更新鸟自身相关信息
    def upDate(self):
        newfit = calcfit(self.x)
        self.fit = newfit
        if newfit > self.bestFit:
            self.bestFit = newfit
            self.bestAddr = self.x.copy()

#种群的类，里面有很多鸟
class Group:
    def __init__(self):
        self.groupSize=500  #鸟的个数、粒子个数
        self.addrSize=48    #位置的维度，也就是TSP城市数量
        self.w=0.25 #w为惯性系数，也就是保留上次速度的程度
        self.c1 = 2
        self.c2 = 2
        self.lb = [-3, 4.1]
        self.ub = [12.1, 5.8]
        self.nvar = 2
        self.initBirds()
        self.best_x, self.best_fit = self.getBest()
        self.Gen=0
    #初始化鸟群
    def initBirds(self):
        self.group=[]
        x = [0 for var in range(self.nvar)]
        for i in range(self.groupSize):
            for item in range(self.nvar):
                x[item] = self.lb[item] + (self.ub[item] - self.lb[item])*random.random()
            bird = Bird(x)
            self.group.append(bird)

    #获取当前离食物最近的鸟
    def getBest(self):
        bestFit=-1
        bestBird=None
        #遍历群体里的所有鸟，找到路径最短的
        for bird in self.group:
            nowfit = calcfit(bird.x)
            if nowfit > bestFit:
                bestFit = nowfit
                bestBird = bird
        return bestBird.x.copy(), bestBird.fit
    #返回所有鸟的距离平均值
    def getAvg(self):
        sum=0
        for p in self.group:
            sum+=p.fit
        return sum/len(self.group)
    #打印最优位置的鸟的相关信息
    def showBest(self):
        print(self.Gen,":",self.best_fit)
    #更新每一只鸟的速度和位置
    def upDateBird(self):
        self.Gen+=1
        for bird in self.group:
            #g代表group，m代表me，分别代表自己和群组最优、自己最优的差
            for xs in range(self.nvar):
                bird.v[xs] = self.w * bird.v[xs] + self.c1*random.random()*(bird.bestAddr[xs] - bird.x[xs]) + self.c2*random.random()*(self.best_x[xs] - bird.x[xs])
                if bird.v[xs] < -bird.vmax:
                    bird.v[xs] = -bird.vmax
                elif bird.v[xs] > bird.vmax:
                    bird.v[xs] = bird.vmax
                bird.x[xs] += bird.v[xs]
                if bird.x[xs] < self.lb[xs]:
                    bird.x[xs] = self.lb[xs]
                elif bird.x[xs] > self.ub[xs]:
                    bird.x[xs] = self.ub[xs]
            bird.fit = calcfit(bird.x)
            #顺便在循环里把最优的鸟更新了，防止二次遍历
            if bird.fit > self.best_fit:
                print('更新：旧fit%g' %self.best_fit)
                self.best_x = bird.x.copy()
                self.best_fit = bird.fit
                print('更新：新fit%g'%self.best_fit)
            bird.upDate()

Gen=[]  #代数
dist=[] #距离
avgDist=[]  #平均距离
#上面三个列表是为了画图
group=Group()
current_iter = 0
#进行若干次迭代
while current_iter < 1000:
    current_iter+=1
    group.upDateBird()
    group.showBest()
    Gen.append(current_iter)
    # dist.append(group.getBest()[1])
    dist.append(group.best_fit)
    # avgDist.append(group.getAvg())
#将过程可视化
plt.plot(Gen,dist,'-r')
# plt.plot(Gen,avgDist,'-b')
plt.show()
print('最优解：', group.best_x, group.best_fit)