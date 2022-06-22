#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
import pandas as pd


# In[97]:


sample_submit = pd.read_csv('sample_submission.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[98]:


train.head()


# In[99]:


train.columns


# In[100]:


train.info()


# In[101]:


#date_timeを文字列ではなく、実際の日時に変換する(train)
train['date_time'] = pd.to_datetime(train['date_time'])


# In[102]:


#date_timeを文字列ではなく、実際の日時に変換する(test)
test['date_time'] = pd.to_datetime(test['date_time'])


# In[103]:


#date_timeから派生変数を作成する(train)
train['date_day'] = train['date_time'].dt.day
train['date_month'] = train['date_time'].dt.month
train['date_year'] = train['date_time'].dt.year
train['date_hour'] = train['date_time'].dt.hour


# In[104]:


#date_timeから派生変数を作成する(test)
test['date_day'] = test['date_time'].dt.day
test['date_month'] = test['date_time'].dt.month
test['date_year'] = test['date_time'].dt.year
test['date_hour'] = test['date_time'].dt.hour


# In[105]:


train.head()


# In[106]:


train.info()


# In[107]:


sample_submit.head()


# In[108]:


train.columns


# In[109]:


cols_carbon = ['date_time',
               'deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'target_carbon_monoxide',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']


# In[110]:


cols_benzene = ['date_time',
               'deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'target_benzene',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']


# In[111]:


cols_nitrogen = ['date_time',
               'deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'target_nitrogen_oxides',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']


# In[112]:


#列変数に対して、トレーニングデータセットの作成
train_carbon = train[cols_carbon]
train_benzene = train[cols_benzene]
train_nitrogen = train[cols_nitrogen]


# In[113]:


train_carbon.head()


# In[114]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
import seaborn as sns


# In[115]:


import pycaret
from pycaret.regression import *


# In[116]:


train.shape


# In[117]:


plt.figure(figsize=(15,15))
sns.heatmap(train.corr(), cmap=plt.cm.Blues, vmin=-1, vmax=1, cbar_kws={'shrink': .8}, square=True, 
            annot=True, fmt='.2f', linewidths=.8)
plt.show()


# In[118]:


fig, ax = plt.subplots(3, 3, figsize = (18,10))

ax[0,0].plot(train.groupby(train['date_month'])['target_carbon_monoxide'].mean(), 'r');
ax[0,1].plot(train.groupby(train['date_month'])['target_benzene'].mean(), 'r');
ax[0,2].plot(train.groupby(train['date_month'])['target_nitrogen_oxides'].mean(), 'r');

ax[1,0].plot(train.groupby(train['date_day'])['target_carbon_monoxide'].mean(), 'y');
ax[1,1].plot(train.groupby(train['date_day'])['target_benzene'].mean(), 'y');
ax[1,2].plot(train.groupby(train['date_day'])['target_nitrogen_oxides'].mean(), 'y');

ax[2,0].plot(train.groupby(train['date_hour'])['target_carbon_monoxide'].mean(), 'b');
ax[2,1].plot(train.groupby(train['date_hour'])['target_benzene'].mean(), 'b');
ax[2,2].plot(train.groupby(train['date_hour'])['target_nitrogen_oxides'].mean(), 'b');

ax[0,0].set_title('Month-Carb')
ax[0,1].set_title('Month-Benz')
ax[0,2].set_title('Month-Nitr')

ax[1,0].set_title('Day-Carb')
ax[1,1].set_title('Day-Benz')
ax[1,2].set_title('Day-Nitr')

ax[2,0].set_title('Hour-Carb')
ax[2,1].set_title('Hour-Benz')
ax[2,2].set_title('Hour-Nitr')

plt.show()


# In[119]:


targets = ["target_carbon_monoxide", "target_benzene", "target_nitrogen_oxides"]
target_names = ["Carbon monoxide", "Benzene", "Nitrogen oxides"]


# In[120]:


fig, axs = plt.subplots(figsize=(15, 6), ncols=3, nrows=1, sharey=False)

fig.suptitle("Target values distribution", fontsize=20)

colors = ["mediumorchid", "lightseagreen", "cornflowerblue"]

for i in [0, 1, 2]:
    axs[i].hist(train[targets[i]], bins=40, edgecolor="black", color=colors[i])
    axs[i].set_title(f"{target_names[i]} (target {i+1})", fontsize=15, pad=5)
    axs[i].set_ylabel("Amount of values", fontsize=13, labelpad=5)
    axs[i].set_xlabel(f"{target_names[i]} level", fontsize=13, labelpad=5)
    axs[i].grid(axis="y")

plt.show()


# In[121]:


fig, axs = plt.subplots(figsize=(15, 6), ncols=3, nrows=1, sharey=False)

fig.suptitle("Target values distribution", fontsize=20)

colors = ["mediumorchid", "lightseagreen", "cornflowerblue"]

for i in [0, 1, 2]:
    axs[i].boxplot(train[targets[i]])
    axs[i].set_title(f"{target_names[i]} (target #{i+1})", fontsize=15, pad=5)
    axs[i].set_ylabel("Amount of values", fontsize=13, labelpad=5)
    axs[i].set_xlabel(f"{target_names[i]} level", fontsize=13, labelpad=5)
    axs[i].grid(axis="y")

plt.show()


# In[122]:


train_carbon.plot(x="date_time", y="target_carbon_monoxide", rot=50)


# In[123]:


train_benzene.plot(x="date_time", y="target_benzene", rot=50)


# In[124]:


train_nitrogen.plot(x="date_time", y="target_nitrogen_oxides", rot=50)


# In[256]:


from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from mlxtend.regressor import StackingCVRegressor
#import lightgbm as lgb
import xgboost as xgb
#import catboost as cbt
import sklearn.metrics as metrics
import sklearn.model_selection as ms
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.metrics import r2_score
from sklearn import preprocessing


# ## GradientBoostingRegressor

# In[126]:


TRAINING_COLUMNS = ['deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',]


# In[129]:


target = train_carbon.target_carbon_monoxide
train_dataset = train_carbon[TRAINING_COLUMNS]
X_train, X_test, y_train, y_test = train_test_split(train_dataset, target, test_size=0.2, random_state=21)

gbr_target_carbon_monoxide = GradientBoostingRegressor()

gbr_target_carbon_monoxide.fit(X_train,y_train)
y_pred = gbr_target_carbon_monoxide.predict(X_test)
print("R2 Score : ",r2_score(y_test, y_pred))
cross_validate(gbr_target_carbon_monoxide, X_train, y_train, cv=6)


# In[130]:


target = train_benzene.target_benzene
train_dataset = train_carbon[TRAINING_COLUMNS]
X_train, X_test, y_train, y_test = train_test_split(train_dataset, target, test_size=0.2, random_state=21)

gbr_target_benzene = GradientBoostingRegressor()

gbr_target_benzene.fit(X_train,y_train)
y_pred = gbr_target_benzene.predict(X_test)
print("R2 Score : ",r2_score(y_test, y_pred))
cross_validate(gbr_target_benzene, X_train, y_train, cv=6)


# In[131]:


target = train_nitrogen.target_nitrogen_oxides
train_dataset = train_nitrogen[TRAINING_COLUMNS]
X_train, X_test, y_train, y_test = train_test_split(train_dataset, target, test_size=0.2, random_state=21)

gbr_target_nitrogen_oxides = GradientBoostingRegressor()

gbr_target_nitrogen_oxides.fit(X_train,y_train)
y_pred = gbr_target_nitrogen_oxides.predict(X_test)
print("R2 Score : ",r2_score(y_test, y_pred))
cross_validate(gbr_target_nitrogen_oxides, X_train, y_train, cv=6)


# In[132]:


test_target_carbon_monoxide = gbr_target_carbon_monoxide.predict(test[TRAINING_COLUMNS])
test_target_benzene = gbr_target_benzene.predict(test[TRAINING_COLUMNS])
test_target_nitrogen_oxides = gbr_target_nitrogen_oxides.predict(test[TRAINING_COLUMNS])


# In[133]:


test.head()


# In[134]:


submission = pd.DataFrame({
    'date_time':date_time,
    'target_carbon_monoxide': test_target_carbon_monoxide.round(1),
    'target_benzene' : test_target_benzene.round(1),
    'target_nitrogen_oxides' : test_target_nitrogen_oxides.round(1)
})
submission.to_csv('submission.csv', index=False)


# Score : 0.26970

# ## RandomForestRegressor

# In[223]:


train_X = train_carbon[['deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']]
train_y = train_carbon['target_carbon_monoxide']


# In[224]:


X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3,random_state=30)


# In[225]:


print('Training shapes:',X_train.shape, y_train.shape)
print('Test shapes:',X_test.shape, y_test.shape)


# In[226]:


RF_carbon_monoxide = RandomForestRegressor(max_depth = 5, n_estimators=100, random_state = 0)
RF_carbon_monoxide.fit(X_train, y_train)
y_pred=RF_carbon_monoxide.predict(X_test)
print(RF_carbon_monoxide.score(X_test,y_test))
print(RF_carbon_monoxide.score(X_train, y_train))
print("R2 Score : ",r2_score(y_test, y_pred))


# In[227]:


rmse_RF_carbon_monoxide=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse_RF_carbon_monoxide)


# In[228]:


train_X = train_benzene[['deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']]
train_y = train_carbon['target_carbon_monoxide']
train_y = train_benzene['target_benzene']

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3,random_state=30)

RF_benzene = RandomForestRegressor(max_depth = 5, n_estimators=100, random_state = 0)
RF_benzene.fit(X_train, y_train)
y_pred=RF_benzene.predict(X_test)
print(RF_benzene.score(X_test,y_test))
print(RF_benzene.score(X_train, y_train))
print("R2 Score : ",r2_score(y_test, y_pred))


# In[229]:


rmse_RF_benzene=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse_RF_benzene)


# In[230]:


train_X = train_nitrogen[['deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']]
train_y = train_nitrogen['target_nitrogen_oxides']

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3,random_state=30)

RF_nitrogen_oxides = RandomForestRegressor(max_depth = 5, n_estimators=100, random_state = 0)
RF_nitrogen_oxides.fit(X_train, y_train)
y_pred=RF_nitrogen_oxides.predict(X_test)
print(RF_nitrogen_oxides.score(X_test,y_test))
print(RF_nitrogen_oxides.score(X_train, y_train))
print("R2 Score : ",r2_score(y_test, y_pred))


# In[231]:


rmse_RF_nitrogen_oxides=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse_RF_nitrogen_oxides)


# In[232]:


test_columns = test[['deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']]

test_target_carbon_monoxide = RF_carbon_monoxide.predict(test_columns)
test_target_benzene = RF_benzene.predict(test_columns)
test_target_nitrogen_oxides = RF_nitrogen_oxides.predict(test_columns)


# In[233]:


submission = pd.DataFrame({
    'date_time':date_time,
    'target_carbon_monoxide': test_target_carbon_monoxide.round(1),
    'target_benzene' : test_target_benzene.round(1),
    'target_nitrogen_oxides' : test_target_nitrogen_oxides.round(1)
})
submission.to_csv('submission2.csv', index=False)


# 0.47925

# ## KNNの正規化

# In[208]:


train_X = train_carbon[['deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']]
train_y = train_carbon['target_carbon_monoxide']


# In[209]:


X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3,random_state=30)


# In[210]:


#MinMaxScalerを訓練データに適応
MM_carbon_monoxide = MinMaxScaler().fit(X_train)

#訓練データを変換する
X_train_MM_carbon = MM_carbon_monoxide.transform(X_train)

#テストデータを変換
X_test_MM_carbon = MM_carbon_monoxide.transform(X_test)


# In[211]:


trainX = [X_train, X_train_MM_carbon]
testX = [X_test, X_test_MM_carbon]


# In[212]:


rmse_carbon_monoxide = []

for i in range(len(trainX)):
    knn_carbon_monoxide = KNeighborsRegressor(n_neighbors=5)
    knn_carbon_monoxide.fit(trainX[i], y_train)
    y_pred = knn_carbon_monoxide.predict(testX[i])
    print(knn_carbon_monoxide.score(testX[i], y_test))
    rmse_carbon_monoxide.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
print(rmse_carbon_monoxide)
print("R2 Score : ",r2_score(y_test, y_pred))


# In[213]:


train_X = train_benzene[['deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']]
train_y = train_benzene['target_benzene']

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3,random_state=30)

#MinMaxScalerを訓練データに適応
MM_benzene = MinMaxScaler().fit(X_train)

#訓練データを変換する
X_train_MM_benzene = MM_benzene.transform(X_train)

#テストデータを変換
X_test_MM_benzene = MM_benzene.transform(X_test)

trainX = [X_train, X_train_MM_benzene]
testX = [X_test, X_test_MM_benzene]


# In[214]:


rmse_benzene = []

for i in range(len(trainX)):
    knn_benzene = KNeighborsRegressor(n_neighbors=5)
    knn_benzene.fit(trainX[i], y_train)
    y_pred = knn_benzene.predict(testX[i])
    print(knn_benzene.score(testX[i], y_test))
    rmse_benzene.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
print(rmse_benzene)
print("R2 Score : ",r2_score(y_test, y_pred))


# In[215]:


train_X = train_nitrogen[['deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']]
train_y = train_nitrogen['target_nitrogen_oxides']

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3,random_state=30)

#MinMaxScalerを訓練データに適応
MM_nitrogen_oxides = MinMaxScaler().fit(X_train)

#訓練データを変換する
X_train_MM_nitrogen = MM_nitrogen_oxides.transform(X_train)

#テストデータを変換
X_test_MM_nitrogen = MM_nitrogen_oxides.transform(X_test)

trainX = [X_train, X_train_MM_nitrogen]
testX = [X_test, X_test_MM_nitrogen]


# In[216]:


rmse_nitrogen_oxides = []

for i in range(len(trainX)):
    knn_nitrogen_oxides = KNeighborsRegressor(n_neighbors=5)
    knn_nitrogen_oxides.fit(trainX[i], y_train)
    y_pred = knn_nitrogen_oxides.predict(testX[i])
    print(knn_nitrogen_oxides.score(testX[i], y_test))
    rmse_nitrogen_oxides.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
print(rmse_nitrogen_oxides)
print("R2 Score : ",r2_score(y_test, y_pred))


# In[221]:


test_columns = test[['deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']]

test_target_carbon_monoxide = knn_carbon_monoxide.predict(test_columns)
test_target_benzene = knn_benzene.predict(test_columns)
test_target_nitrogen_oxides = knn_nitrogen_oxides.predict(test_columns)


# In[222]:


submission = pd.DataFrame({
    'date_time':date_time,
    'target_carbon_monoxide': test_target_carbon_monoxide.round(1),
    'target_benzene' : test_target_benzene.round(1),
    'target_nitrogen_oxides' : test_target_nitrogen_oxides.round(1)
})
submission.to_csv('submission1.csv', index=False)


# 1.63539

# ## GradientBoostingRegressor ver2

# In[306]:


training_columns =  ['deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']


# In[307]:


target = train_carbon.target_carbon_monoxide
train_dataset = train_carbon[training_columns]
X_train, X_test, y_train, y_test = train_test_split(train_dataset, target, test_size=0.2, random_state=21)

gbr_target_carbon_monoxide = GradientBoostingRegressor()

gbr_target_carbon_monoxide.fit(X_train,y_train)
y_pred = gbr_target_carbon_monoxide.predict(X_test)
print("R2 Score : ",r2_score(y_test, y_pred))
cross_validate(gbr_target_carbon_monoxide, X_train, y_train, cv=6)


# In[308]:


target = train_benzene.target_benzene
train_dataset = train_carbon[training_columns]
X_train, X_test, y_train, y_test = train_test_split(train_dataset, target, test_size=0.2, random_state=21)

gbr_target_benzene = GradientBoostingRegressor()

gbr_target_benzene.fit(X_train,y_train)
y_pred = gbr_target_benzene.predict(X_test)
print("R2 Score : ",r2_score(y_test, y_pred))
cross_validate(gbr_target_benzene, X_train, y_train, cv=6)


# In[309]:


target = train_nitrogen.target_nitrogen_oxides
train_dataset = train_nitrogen[training_columns]
X_train, X_test, y_train, y_test = train_test_split(train_dataset, target, test_size=0.2, random_state=21)

gbr_target_nitrogen_oxides = GradientBoostingRegressor()
gbr_target_nitrogen_oxides.fit(X_train,y_train)
y_pred = gbr_target_nitrogen_oxides.predict(X_test)
print("R2 Score : ",r2_score(y_test, y_pred))
print(gbr_target_nitrogen_oxides.score(X_test,y_test))
print(gbr_target_nitrogen_oxides.score(X_train, y_train))
cross_validate(gbr_target_nitrogen_oxides, X_train, y_train, cv=6)


# In[247]:


test_columns = ['deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']

test_target_carbon_monoxide = gbr_target_carbon_monoxide.predict(test[test_columns])
test_target_benzene = gbr_target_benzene.predict(test[test_columns])
test_target_nitrogen_oxides = knn_nitrogen_oxides.predict(test[test_columns])


# In[248]:


submission = pd.DataFrame({
    'date_time':date_time,
    'target_carbon_monoxide': test_target_carbon_monoxide.round(1),
    'target_benzene' : test_target_benzene.round(1),
    'target_nitrogen_oxides' : test_target_nitrogen_oxides.round(1)
})
submission.to_csv('submission3.csv', index=False)


# 0.39241

# ## 回帰モデル評価

# In[293]:


target = train_carbon.target_carbon_monoxide
train_dataset = train_carbon[training_columns]
X_train, X_test, y_train, y_test = train_test_split(train_dataset, target, test_size=0.5, random_state=0)


#標準化
sc_target_carbon_monoxide = StandardScaler()
sc_target_carbon_monoxide.fit(X_train)
X_train_carbon_monoxide = sc_target_carbon_monoxide.transform(X_train)
X_test_carbon_monoxide = sc_target_carbon_monoxide.transform(X_test)


# In[294]:


# モデルの設定
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=0),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=0),
    'LinearSVR': LinearSVR(random_state=0)
}

# 評価値の計算
scores = {}
for model_name, model in models.items():
    model.fit(X_train_carbon_monoxide, y_train)
    scores[(model_name, 'MSE')] = mean_squared_error(y_test, model.predict(X_test_carbon_monoxide))
    scores[(model_name, 'RMSE')] = np.sqrt(mean_squared_error(y_test, model.predict(X_test_carbon_monoxide)))
    scores[(model_name, 'MAE')] = mean_absolute_error(y_test, model.predict(X_test_carbon_monoxide))
    scores[(model_name, 'MedAE')] = median_absolute_error(y_test, model.predict(X_test_carbon_monoxide))
    scores[(model_name, 'R2')] = r2_score(y_test, model.predict(X_test_carbon_monoxide))

#表示
pd.Series(scores).unstack()


# In[295]:


target = train_benzene.target_benzene
train_dataset = train_benzene[training_columns]
X_train, X_test, y_train, y_test = train_test_split(train_dataset, target, test_size=0.5, random_state=0)


#標準化
sc_target_benzene = StandardScaler()
sc_target_benzene.fit(X_train)
X_train_benzene = sc_target_benzene.transform(X_train)
X_test_benzene = sc_target_benzene.transform(X_test)


# In[296]:


# モデルの設定
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=0),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=0),
    'LinearSVR': LinearSVR(random_state=0)
}

# 評価値の計算
scores = {}
for model_name, model in models.items():
    model.fit(X_train_benzene, y_train)
    scores[(model_name, 'MSE')] = mean_squared_error(y_test, model.predict(X_test_benzene))
    scores[(model_name, 'RMSE')] = np.sqrt(mean_squared_error(y_test, model.predict(X_test_benzene)))
    scores[(model_name, 'MAE')] = mean_absolute_error(y_test, model.predict(X_test_benzene))
    scores[(model_name, 'MedAE')] = median_absolute_error(y_test, model.predict(X_test_benzene))
    scores[(model_name, 'R2')] = r2_score(y_test, model.predict(X_test_benzene))

#表示
pd.Series(scores).unstack()


# In[297]:


target = train_nitrogen.target_nitrogen_oxides
train_dataset = train_nitrogen[training_columns]
X_train, X_test, y_train, y_test = train_test_split(train_dataset, target, test_size=0.5, random_state=0)


#標準化
sc_target_nitrogen_oxides = StandardScaler()
sc_target_nitrogen_oxides.fit(X_train)
X_train_nitrogen_oxides = sc_target_nitrogen_oxides.transform(X_train)
X_test_nitrogen_oxides = sc_target_nitrogen_oxides.transform(X_test)


# In[298]:


# モデルの設定
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=0),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=0),
    'LinearSVR': LinearSVR(random_state=0)
}

# 評価値の計算
scores = {}
for model_name, model in models.items():
    model.fit(X_train_nitrogen_oxides, y_train)
    scores[(model_name, 'MSE')] = mean_squared_error(y_test, model.predict(X_test_nitrogen_oxides))
    scores[(model_name, 'RMSE')] = np.sqrt(mean_squared_error(y_test, model.predict(X_test_nitrogen_oxides)))
    scores[(model_name, 'MAE')] = mean_absolute_error(y_test, model.predict(X_test_nitrogen_oxides))
    scores[(model_name, 'MedAE')] = median_absolute_error(y_test, model.predict(X_test_nitrogen_oxides))
    scores[(model_name, 'R2')] = r2_score(y_test, model.predict(X_test_nitrogen_oxides))

#表示
pd.Series(scores).unstack()


# ## 決定木(nitrogen)

# In[284]:


target = train_nitrogen.target_nitrogen_oxides
train_dataset = train_nitrogen[training_columns]
X_train, X_test, y_train, y_test = train_test_split(train_dataset, target, test_size=0.5, random_state=0)


#標準化
sc_target_nitrogen_oxides = StandardScaler()
sc_target_nitrogen_oxides.fit(X_train)
X_train_nitrogen_oxides = sc_target_nitrogen_oxides.transform(X_train)
X_test_nitrogen_oxides = sc_target_nitrogen_oxides.transform(X_test)


# In[285]:


DTR_target_nitrogen_oxides = DecisionTreeRegressor(random_state=0)
DTR_target_nitrogen_oxides.fit(X_train_nitrogen_oxides, y_train)
#y_pred = DTR_nitrogen_oxides.predict((X_test_nitrogen_oxides)
DTR_target_nitrogen_oxides.score(X_test_nitrogen_oxides, y_test)
#rmse_nitrogen_oxides.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
#print(rmse_nitrogen_oxides)
#print("R2 Score : ",r2_score(y_test, y_pred))


# ## LinearRegression(carbon)

# In[286]:


target = train_carbon.target_carbon_monoxide
train_dataset = train_carbon[training_columns]
X_train, X_test, y_train, y_test = train_test_split(train_dataset, target, test_size=0.5, random_state=0)


#標準化
sc_target_carbon_monoxide = StandardScaler()
sc_target_carbon_monoxide.fit(X_train)
X_train_carbon_monoxide = sc_target_carbon_monoxide.transform(X_train)
X_test_carbon_monoxide = sc_target_carbon_monoxide.transform(X_test)


# In[287]:


LR_target_carbon_monoxide = LinearRegression()

LR_target_carbon_monoxide.fit(X_train_carbon_monoxide ,y_train)
LR_target_carbon_monoxide.score(X_test_carbon_monoxide, y_test)
#y_pred = gbr_target_carbon_monoxide.predict(X_test)
#print("R2 Score : ",r2_score(y_test, y_pred))
#cross_validate(gbr_target_carbon_monoxide, X_train, y_train, cv=6)


# ## Ridge(benzene)

# In[288]:


target = train_benzene.target_benzene
train_dataset = train_benzene[training_columns]
X_train, X_test, y_train, y_test = train_test_split(train_dataset, target, test_size=0.5, random_state=0)


#標準化
sc_target_benzene = StandardScaler()
sc_target_benzene.fit(X_train)
X_train_benzene = sc_target_benzene.transform(X_train)
X_test_benzene = sc_target_benzene.transform(X_test)


# In[289]:


ridge_target_benzene = Ridge(random_state=0)

ridge_target_benzene.fit(X_train_benzene, y_train)
ridge_target_benzene.score(X_test_benzene, y_test)


# In[290]:


test_columns = ['deg_C', 
               'relative_humidity', 
               'absolute_humidity',
               'sensor_1', 
               'sensor_2', 
               'sensor_3', 
               'sensor_4', 
               'sensor_5',
               'date_day', 
               'date_month', 
               'date_year', 
               'date_hour']

test_target_carbon_monoxide = LR_target_carbon_monoxide.predict(test[test_columns])
test_target_benzene = ridge_target_benzene.predict(test[test_columns])
test_target_nitrogen_oxides = DTR_target_nitrogen_oxides.predict(test[test_columns])


# In[291]:


submission = pd.DataFrame({
    'date_time':date_time,
    'target_carbon_monoxide': test_target_carbon_monoxide.round(1),
    'target_benzene' : test_target_benzene.round(1),
    'target_nitrogen_oxides' : test_target_nitrogen_oxides.round(1)
})
submission.to_csv('submission0.csv', index=False)


# 5.17580

# In[ ]:





# In[299]:


test_target_carbon_monoxide = LR_target_carbon_monoxide.predict(test[test_columns])
test_target_benzene = ridge_target_benzene.predict(test[test_columns])
test_target_nitrogen_oxides = RF_nitrogen_oxides.predict(test[test_columns])


# In[300]:


submission = pd.DataFrame({
    'date_time':date_time,
    'target_carbon_monoxide': test_target_carbon_monoxide.round(1),
    'target_benzene' : test_target_benzene.round(1),
    'target_nitrogen_oxides' : test_target_nitrogen_oxides.round(1)
})
submission.to_csv('submission4.csv', index=False)


# スコア：5.06253

# In[ ]:




