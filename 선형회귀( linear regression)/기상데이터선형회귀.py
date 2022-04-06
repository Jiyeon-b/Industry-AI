import pandas as pd
import numpy as np
import statsmodels.api as sm 
import matplotlib.pyplot as plt

#데이터 불러오기
data = pd.read_csv("./2020.csv", encoding='euc-kr')
 
#데이터 통계 보기
print(data.describe())

#타겟 설정
target = data[['renergy']]
temp = data[['temp']]
humi = data[['humi']]
dust = data[['dust']]
solar = data[['solar']]
solartime = data[['solartime']]
groundtemp = data[['groundtemp']]
totalcloud = data[['totalcloud']]

 
#temp변수에 상수항추가하기
temp1 = sm.add_constant(temp,has_constant="add")
 
#sm.OLS 적합시키기
tempmodel = sm.OLS(target,temp1)
fitted_model_temp = tempmodel.fit()
 
#summary 함수통해 결과 출력
print(fitted_model_temp.summary())
 
#회귀 계수 출력
#print(fitted_model1.params)

#회귀 계수 x 데이터(x)
np.dot(temp1,fitted_model_temp.params)
pred1 = fitted_model_temp.predict()

#적합시킨 직선 시각화
plt.figure(figsize=(10,8))
plt.scatter(temp,target,label="data")
plt.plot(temp,pred1,label="result",color="red")
plt.legend()
plt.xlabel('temp', fontsize=15)
plt.ylabel('renergy',fontsize=15)
plt.show()

#residual 시각화
plt.figure(figsize=(10,8))
fitted_model_temp.resid.plot()
plt.xlabel("temp_residual_number")
plt.show()

#잔차의 합계산해보기
print("잔차 : ",np.sum(fitted_model_temp.resid))

#위와 동일하게 모든변수로 각각 단순선형회귀분석 결과보기

#상수항추가
solar1 = sm.add_constant(solar,has_constant="add")
totalcloud1 = sm.add_constant(totalcloud,has_constant="add")
solartime1 = sm.add_constant(solartime,has_constant="add")
humi1 = sm.add_constant(humi,has_constant="add")
dust1 = sm.add_constant(dust,has_constant="add")
groundtemp1 = sm.add_constant(groundtemp,has_constant="add")

#회귀모델 적합
solarmodel = sm.OLS(target,solar1)
fitted_model_solar = solarmodel.fit()

totalcloudmodel = sm.OLS(target,totalcloud1)
fitted_model_totalcloud = totalcloudmodel.fit()

solartimemodel = sm.OLS(target,solartime1)
fitted_model_solartime = solartimemodel.fit()

humimodel = sm.OLS(target, humi1)
fitted_model_humi = humimodel.fit()

dustmodel = sm.OLS(target,dust1)
fitted_model_dust = dustmodel.fit()

groundtempmodel = sm.OLS(target,groundtemp1)
fitted_model_groundtemp = groundtempmodel.fit()

#solar 모델결과출력
fitted_model_solar.summary()
#totalcloud1 모델결과출력
fitted_model_totalcloud.summary()
#solartime 모델결과출력
fitted_model_solartime.summary()
#humi 모델결과출력
fitted_model_humi.summary()
#dust 모델결과출력
fitted_model_dust.summary()
#groundtemp1 모델결과출력
fitted_model_groundtemp.summary()

# 각각 yhat_예측하기
predsolar = fitted_model_solar.predict(solar1)
predtotalcloud = fitted_model_totalcloud.predict(totalcloud1)
predsolartime = fitted_model_solartime.predict(solartime1)
predhumi = fitted_model_humi.predict(humi1)
preddust = fitted_model_dust.predict(dust1)
predgroundtemp = fitted_model_groundtemp.predict(groundtemp1)

#solar 시각화
plt.figure(figsize=(10,8))
plt.scatter(solar,target,label="data")
plt.plot(solar,predsolar,label="result",color="red")
plt.legend()
plt.xlabel('solar', fontsize=15)
plt.ylabel('renergy',fontsize=15)
plt.show()

#totalcloud 시각화
plt.figure(figsize=(10,8))
plt.scatter(totalcloud,target,label="data")
plt.plot(totalcloud,predtotalcloud,label="result",color="red")
plt.legend()
plt.xlabel('totalcloud', fontsize=15)
plt.ylabel('renergy',fontsize=15)
plt.show()

#solartime 시각화
plt.figure(figsize=(10,8))
plt.scatter(solartime,target,label="data")
plt.plot(solartime,predsolartime,label="result",color="red")
plt.legend()
plt.xlabel('solartime', fontsize=15)
plt.ylabel('renergy',fontsize=15)
plt.show()

#humi 시각화
plt.figure(figsize=(10,8))
plt.scatter(humi,target,label="data")
plt.plot(humi,predhumi,label="result",color="red")
plt.legend()
plt.xlabel('humi', fontsize=15)
plt.ylabel('renergy',fontsize=15)
plt.show()

#dust 시각화
plt.figure(figsize=(10,8))
plt.scatter(dust,target,label="data")
plt.plot(dust,preddust,label="result",color="red")
plt.legend()
plt.xlabel('dust', fontsize=15)
plt.ylabel('renergy',fontsize=15)
plt.show()

#groundtemp 시각화
plt.figure(figsize=(10,8))
plt.scatter(groundtemp,target,label="data")
plt.plot(groundtemp,predgroundtemp,label="result",color="red")
plt.legend()
plt.xlabel('groundtemp', fontsize=15)
plt.ylabel('renergy',fontsize=15)
plt.show()

#solar모델 residual시각화
plt.figure(figsize=(10,8))
fitted_model_solar.resid.plot()
plt.xlabel("solar_residual_number")
plt.show()

#totalcloud모델 residual시각화
plt.figure(figsize=(10,8))
fitted_model_totalcloud.resid.plot()
plt.xlabel("totalcloud_residual_number")
plt.show()

#solartime모델 residual시각화
plt.figure(figsize=(10,8))
fitted_model_solartime.resid.plot()
plt.xlabel("solartime_residual_number")
plt.show()

#humi모델 residual시각화
plt.figure(figsize=(10,8))
fitted_model_humi.resid.plot()
plt.xlabel("humi_residual_number")
plt.show()

#dust모델 residual시각화
plt.figure(figsize=(10,8))
fitted_model_dust.resid.plot()
plt.xlabel("dust_residual_number")
plt.show()

#groundtemp모델 residual시각화
plt.figure(figsize=(10,8))
fitted_model_groundtemp.resid.plot()
plt.xlabel("groundtemp_residual_number")
plt.show()

# 세 모델씩 residual 비교
plt.figure(figsize=(10,8))
fitted_model_temp.resid.plot(label="solar")
fitted_model_totalcloud.resid.plot(label="solartime")
fitted_model_solar.resid.plot(label="totalcloud")
plt.legend()
plt.show()
