from sklearn import linear_model   #导入线性模块
import matplotlib.pyplot as plt
import numpy as np
regression=linear_model.LinearRegression(fit_intercept=True) #创建线性回归模型
x=[[1],[3],[4],[7],[8]]
y=[5,2,8,1,9]
regression.fit(x,y)              #拟合
k=regression.coef_               #获取斜率
b=regression.intercept_          #获取截距
print(k,b)
x0=np.arange(0,10,1)
y=k*x+b
plt.scatter(x,y)
plt.plot(x,y)
plt.show()
a=regression.predict([[6],[99]])   #模型预测未知点
print(a)
b=regression.score([[6],[7],[8],[9],[10]],[1.6,1.8,1.9,10,111])
#对模型进行评分,结果越大越好
print(b)

