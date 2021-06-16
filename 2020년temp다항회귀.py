import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# define random jitter
def rjitt(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

data = pd.read_csv("./2020.csv", encoding='euc-kr')

print(data.columns)
data.head()

plt.scatter(data['temp'], data['renergy'], s=40, alpha=0.5)
plt.title('temp-renergy')
plt.xlabel('temp')
plt.ylabel('renergy')
plt.show()


x = data['temp']
y = data['renergy']
# 회귀함수의 차수 지정: 차수가 1이면 단순 선형회귀분석
nDegree = 1

model = make_pipeline(PolynomialFeatures(nDegree), LinearRegression())
model.fit(np.array(x).reshape(-1, 1), y)
x_reg = np.arange(7)[1:]
y_reg = model.predict(x_reg.reshape(-1, 1)) ## 차원 증가 시켜준다. statsmodels과 달리 설명변수의 차원을 하나 증가시켜야 함

plt.scatter(rjitt(data['temp']), rjitt(data['renergy']),
           s=30, alpha=0.5)
plt.xlabel('temp')
plt.ylabel('renergy')

plt.plot(x_reg, y_reg, color='red') 
plt.show()

x = data['temp']
y = data['renergy']
# 회귀함수의 차수 지정: 3
nDegree = 3

model = make_pipeline(PolynomialFeatures(nDegree), LinearRegression())
model.fit(np.array(x).reshape(-1, 1), y)
x_reg = np.arange(25)[1:]
y_reg = model.predict(x_reg.reshape(-1, 1))

plt.scatter(rjitt(data['temp']), rjitt(data['renergy']),
           s=30, alpha=0.5)
plt.xlabel('temp')
plt.ylabel('renergy')

plt.plot(x_reg, y_reg, color='red')
plt.show()

x = data['temp']
y = data['renergy']
# 회귀함수의 차수 지정: 10
nDegree = 10

model = make_pipeline(PolynomialFeatures(nDegree), LinearRegression())
model.fit(np.array(x).reshape(-1, 1), y)
x_reg = np.arange(25)[1:]
y_reg = model.predict(x_reg.reshape(-1, 1))

plt.scatter(rjitt(data['temp']), rjitt(data['renergy']),
           s=30, alpha=0.5)
plt.xlabel('temp')
plt.ylabel('renergy')

plt.plot(x_reg, y_reg, color='red')
plt.show()


