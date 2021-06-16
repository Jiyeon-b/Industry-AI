from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import pandas as pd

#데이터 불러오기
data = pd.read_csv("./2020.csv", encoding='euc-kr')

#타겟 설정
y = data[['renergy']]
X = data[['temp']]


model = Pipeline([("poly", PolynomialFeatures()),
                  ("lreg", LinearRegression())])

degrees = np.arange(1, 15)
train_scores, test_scores = validation_curve(
    model, X, y, "poly__degree", degrees, cv=100,
    scoring="neg_mean_squared_error")

plt.plot(degrees, test_scores.mean(axis=1), "o-", label="test-score avg")
plt.plot(degrees, train_scores.mean(axis=1), "o--", label="train-score avg")
plt.ylabel('performance')
plt.xlabel('degree')
plt.legend()
plt.title("Temp Normalization")
plt.show()