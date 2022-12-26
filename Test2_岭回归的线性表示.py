#放弃最小二乘法的无偏性，以损失部分信息、降低精度为代价，获得更为实际的可靠的回归系数
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np
x0=np.arange(20)
ridgeRegression1=Ridge(alpha=10)
x=[[3],[8]]
y=[1,2]
ridgeRegression1.fit(x,y)
k1=ridgeRegression1.coef_      #查看回归系数
b1=ridgeRegression1.intercept_ #查看截距
t1=ridgeRegression1.predict([[20]])
print(k1,b1,t1)
y1=k1*x0+b1

ridgeRegression2=Ridge(alpha=-1)
ridgeRegression2.fit(x,y)
k2=ridgeRegression2.coef_      #查看回归系数
b2=ridgeRegression2.intercept_ #查看截距
t2=ridgeRegression2.predict([[20]])
print(k2,b2,t2)
y2=k2*x0+b2

ridgeRegression3=Ridge(alpha=0) #等价于线性回归
ridgeRegression3.fit(x,y)
k3=ridgeRegression3.coef_      #查看回归系数
b3=ridgeRegression3.intercept_ #查看截距
t3=ridgeRegression3.predict([[20]])
print(k3,b3,t3)
y3=k3*x0+b3
plt.plot(x0,y1,'ro-',x0,y2,'k*:',x0,y3,'b^--')
plt.show()




# #岭迹图认为,线条交叉越多,则说明特征之间的多重共线性越高，
# # 应该选择系数较为平稳的喇叭口所对应的α取值作为最佳的正则化参数的取值
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import  linear_model
#
# #设置中文及字符正常显示
# plt.rcParams['font.sans-serif'] = 'SimHei'
# plt.rcParams['axes.unicode_minus'] = False
#
# # 创造10*10的希尔伯特矩阵
# x = 1. / (np.arange(1,11) + np.arange(0,10)[:,np.newaxis])
# y = np.ones(10)
#
# # 计算横坐标
# n_alphas = 200
# alphas = np.logspace(-10,-2,n_alphas)
#
# # 建模。获取每一个正则化去之下的系数组合
# coefs = []
# for a in alphas:
#     ridge = linear_model.Ridge(alpha=a,fit_intercept=False)
#     ridge.fit(x,y)
#     coefs.append(ridge.coef_)
#
# # 可视化
# # 当前的图表和子图可以使用plt.gcf()和plt.gca()获得，分别表示Get Current Figure和Get Current Axes。
# # 在pyplot模块中，许多函数都是对当前的Figure或Axes对象进行处理，比如说：plt.plot()实际上会通过plt.gca()获得当前
# # 的Axes对象ax，然后再调用ax.plot()方法实现真正的绘图。
# ax = plt.gca()
# ax.plot(alphas,coefs)
# ax.set_xscale('log') # 将 y轴 和 x轴 的比例设置为对数比例
# ax.set_xlim(ax.get_xlim()[::-1]) # 将横坐标逆转，再设置坐标轴尺度范围
# plt.xlabel('正则化参数alpha')
# plt.ylabel('系数w')
# plt.title('岭回归下的岭迹图')
# plt.axis('tight')
# plt.show()
