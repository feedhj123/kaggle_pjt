> # KAGGLE STUDY
## Predict Future Sales

- Description: 러시아 최대 소프트웨어 회사중 하나인 1c company의 일일 판매 데이터로 구성된 시계열 데이터 세트이며, 요구하는 조건은 다음 달에 모든 제품과 매장의 총 매출을 예측해 달라는 것이다.

- Evaluation: 제출은 평균 제곱근 오차(RMSE)로 평가되며, 실제 대상 값은 [0,20] 범위로 잘라진다.

### File descriptions

- sales_train.csv : the training set. Daily historical data from January 2013 to October 2015.

- test.csv : the test set. You need to forecast the sales for these shops and products for November 2015.

- sample_submission.csv : a sample submission file in the correct format.

- items.csv : supplemental information about the items/products.

- item_categories.csv : supplemental information about the items categories.

- shops.csv : supplemental information about the shops.

### Data fileds

- ID : an Id that represents a (Shop, Item) tuple within the test set

- shop_id : unique identifier of a shop

- item_id : unique identifier of a product

- item_category_id : unique identifier of item category

- item_cnt_day : number of products sold. You are predicting a monthly amount of this measure

- item_price : current price of an item

- date : date in format dd/mm/yyyy

- date_block_num : a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33

- item_name : name of item

- shop_name : name of shop

- item_category_name : name of item category


>># 코드 미리 뜯어보기 / 학습
- 참조
- [xgboost](https://www.kaggle.com/code/gordotron85/future-sales-xgboost-top-3)
- [LightGBM](https://www.kaggle.com/code/werooring/top-3-5-lightgbm-with-feature-engineering)

## Part1 데이터 준비(데이터 셋 로드)

```python
#모듈 불러오기
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# 데이터셋 가져오기. 
items = pd.read_csv("/content/items.csv")
shops = pd.read_csv("/content/shops.csv")
train = pd.read_csv("/content/sales_train.csv")
test = pd.read_csv("/content/test.csv")
cats = pd.read_csv("/content/item_categories.csv")      
# 연습을 코랩에서 진행했기 때문에 csv주소가 원본과는 조금 다름. 
```

## Part2 - EDA/Preprocessing
### 1. 데이터 결측치 확인
```python
train.isnull().sum()
cats.isnull().sum()
items.isnull().sum()
shops.isnull().sum()
# 데이터 결측치 확인
```
### 2. Data cleaning(이상치 제거)
```python
plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
flierprops = dict(marker='o', markerfacecolor='purple', markersize=6,
                  linestyle='none', markeredgecolor='black')
sns.boxplot(x=train.item_cnt_day, flierprops=flierprops)
# 일자별 제품 판매량 데이터 시각화

plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
sns.boxplot(x=train.item_price, flierprops=flierprops)
# 제품 가격 시각화

# 눈에 띄는 이상치 제거
train = train[(train.item_price < 300000 )& (train.item_cnt_day < 1000)]
## 제품 가격 300000 넘거나 제품 일판매량이 1000이넘는 데이터는 배제함
train = train[train.item_price > 0].reset_index(drop = True)
train.loc[train.item_cnt_day < 1, "item_cnt_day"] = 0
## 제품 가격이 0보다 작거나 일판매량이 1보다 작은 데이터를 배제함
```

### 3.  shop 데이터 정리.

```python
# Якутск Орджоникидзе, 56
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11

## 원본 csv파일을 보면 상기의 주석 처리된 가게 이름이 중첩되는 것을 확인 할 수 있다. 이러한 중첩 데이터를 처리해준다.

shops["city"] = shops.shop_name.str.split(" ").map( lambda x: x[0] )
## 가게 이름에서 도시를 나타내는 부분을 세분화 해줌 여기서 람다함수 x[0]은 가게이름에서 도시 부분을 나타냄 


shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1] )
## 가게이름에서 가게의 종류?(실제 가게이름에서 category에해당하는 부분을 번역기 돌려봤으나 애매함)를 따로 


shops.loc[shops.city == "!Якутск", "city"] = "Якутск"
## 도시이름에서 특수문자가 들어간 부분 특수문자를 없애줌.


category = []
for cat in shops.category.unique():
    if len(shops[shops.category == cat]) >= 5:
        category.append(cat)
shops.category = shops.category.apply( lambda x: x if (x in category) else "other" )
## 카테고리의 행값들중 중첩된 값이 5개 이상인 것들만 그 범주를 유지하고 나머지 것들은 other로 처리해줌.


from sklearn.preprocessing import LabelEncoder
shops["shop_category"] = LabelEncoder().fit_transform(shops.category)
shops["shop_city"] = LabelEncoder().fit_transform(shops.city)
shops = shops[["shop_id", "shop_category", "shop_city"]]
## 라벨 인코딩 통해서 범주형 데이터들 변환해서 저장해줌.
```

### 4. item데이터 정리
```python
import re
def name_correction(x):
    x = x.lower() # all letters lower case 문자들 다 소문자로 변경 
    x = x.partition('[')[0] # partition by square brackets
    x = x.partition('(')[0] # partition by curly brackets 
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x) # remove special characters 정규표현식인데 공부 좀 해봐야할듯
    x = x.replace('  ', ' ') # replace double spaces with single spaces 띄어쓰기 2번 한거 1번으로 수정
    x = x.strip() # remove leading and trailing white space 공백제거
    return x
## 이름을 알맞은 데이터로 변환해줄 함수 선언



items["name1"], items["name2"] = items.item_name.str.split("[", 1).str
items["name1"], items["name3"] = items.item_name.str.split("(", 1).str
## split item names by first bracket 항목 이름을 첫번째 대괄호 및 중괄호로 분할 


items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
## replace special characters and turn to lower case 특수문자를 교체하고 소문자로 전환

 
items = items.fillna('0')
items["item_name"] = items["item_name"].apply(lambda x: name_correction(x))

## fill nulls with '0' 결측치를 0으로 채우기, item_name 전체에 위에서 지정한 name_correction 함수 적용


items.name2 = items.name2.apply( lambda x: x[:-1] if x !="0" else "0")
## name2가 "0"이 아닌 경우 마지막을 제외한 모든 문자 반환 
```