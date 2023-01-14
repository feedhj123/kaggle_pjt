> # Predict Future Sales

## 데이터와 모듈 불러오기

```python
# 기본적인 모듈만 우선 import
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

# dataset 불러오기
items = pd.read_csv("/content/items.csv")
shops = pd.read_csv("/content/shops.csv")
train = pd.read_csv("/content/sales_train.csv")
test = pd.read_csv("/content/test.csv")
cats = pd.read_csv("/content/item_categories.csv")

```
## 결측치 확인
```python
train.isnull().sum()
shops.isnull().sum()
items.isnull().sum()
test.isnull().sum()
cats.isnull().sum()

train.info()
# 결측치는 없었음
```

## 이상치 확인 및 제거

```python
plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
flierprops = dict(marker='o', markerfacecolor='purple', markersize=6,
                  linestyle='none', markeredgecolor='black')
sns.boxplot(x=train.item_cnt_day, flierprops=flierprops)

plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
sns.boxplot(x=train.item_price, flierprops=flierprops)
# 데이터 분포 시각화하여 우선 확인

#로그로 변환
train['item_price'] = train['item_price'].clip(lower=1)
train['ln_price'] = np.round(np.log(train['item_price'])*10).astype('str')
#실수형 타입으로 재변환
train['ln_price'] = train['ln_price'].astype(float)

#z_score적용
mean = np.mean(train.ln_price,axis = 0)
std = np.std(train.ln_price,axis =0)
train.ln_price = (train.ln_price - mean)/std
# 시각화
sns.displot(data=train, x='ln_price',kde = True)

import seaborn as sns

#z_score 기준으로 편차가 큰 값들 제거
filtered_train = train.loc[(train['ln_price'] < 2.5) & (train['ln_price'] > -2.5)]


sns.displot(data=filtered_train, x='ln_price', kde=True)


# 원본에서도 동일한 범위로 이상치로 판단된 데이터를 한번 지우고 확인을해봤는데
데이터 손실이 너무 많이 발생합니다(5만개이상).

## 범위를 -5~5로 지정해줬더니 손실량이 1000개가량으로 줄어들긴함.
```
