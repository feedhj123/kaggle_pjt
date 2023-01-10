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



items["type"] = items.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )
items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"
## xbox 360으로 통일시킴
items.loc[ items.type == "", "type"] = "mac"
## 빈공간 mac으로 채움
items.type = items.type.apply( lambda x: x.replace(" ", "") )
## 공백 제거해줌
items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc"
## pc로 전체 통일
items.loc[ items.type == 'рs3' , "type"] = "ps3"
## ps3로 통일

group_sum = items.groupby(["type"]).agg({"item_id": "count"})
# groupby.agg -> 함수 여러개 사용하거나 할 떄 사용 이경우에는 타입에 따른 같은 제품의 개 수를 더한것
group_sum = group_sum.reset_index()
drop_cols = []
for cat in group_sum.type.unique():
    if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40:
        drop_cols.append(cat)
# 제품 개수가 40개 이하인 것은 drop_cols에 저장
items.name2 = items.name2.apply( lambda x: "other" if (x in drop_cols) else x )
# 40개 이하인 것들은 other로 지정 이외에는 제품명 그대로 사용
items = items.drop(["type"], axis = 1)
# type 열은 지워줌



items.name2 = LabelEncoder().fit_transform(items.name2)
items.name3 = LabelEncoder().fit_transform(items.name3)

items.drop(["item_name", "name1"],axis = 1, inplace= True)
items.head()
## 라벨 인코딩 후, 기존 제품명과 제품1을 삭제해줌 
```

### 5.Preprocessing

```python
from itertools import product
import time
# - 월, 상점 및 품목의 모든 조합을 증가하는 월 순으로 행렬 df를 만듭니다. Item_cnt_day는 Item_cnt_month로 합계됩니다.
ts = time.time()
matrix = []
cols  = ["date_block_num", "shop_id", "item_id"]
for i in range(34):
    sales = train[train.date_block_num == i]
    # bull indexing
    matrix.append( np.array(list( product( [i], sales.shop_id.unique(), sales.item_id.unique() ) ), dtype = np.int16) )
    # 왜 이걸 만들었을까??

matrix = pd.DataFrame( np.vstack(matrix), columns = cols )
# vstack은 배열을 세로 결합하는 함수
matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8)
# 매트릭스 date block num 속성을 정수로 변환해줌
matrix["shop_id"] = matrix["shop_id"].astype(np.int8)
matrix["item_id"] = matrix["item_id"].astype(np.int16)
# 위와 마찬가지로 정수변환
matrix.sort_values( cols, inplace = True )
#오름차순으로 정렬해줌(위에서 정수로 타입 변경해준 컬럼들)
time.time()- ts



train["revenue"] = train["item_cnt_day"] * train["item_price"]
# 수입 열을 따로 만들어줌 수입 = 제품 가격 x 제품 판매량
ts = time.time()
group = train.groupby( ["date_block_num", "shop_id", "item_id"] ).agg( {"item_cnt_day": ["sum"]} )
# 월별,가게이름,제품명에 따른 판매량의 합계를 구해줌
group.columns = ["item_cnt_month"]
# 새로운 컬럼을 만들어준다.
group.reset_index( inplace = True)
matrix = pd.merge( matrix, group, on = cols, how = "left" )
# https://wikidocs.net/153875 merge 함수 설명 
# how: 병합시 기준이 될 인덱스를 정하는 방식
# on: 열 기준 병합시 기준으로할 열의 이름이 양측이름이 다르면 어떤 열을 기준으로할지 정해준다.
matrix["item_cnt_month"] = matrix["item_cnt_month"].fillna(0).astype(np.float16)
# 매트릭스의 item_cnt_month의 결측치를 0으로 채워줌
time.time() - ts


# Create a test set for month 34.

test["date_block_num"] = 34
test["date_block_num"] = test["date_block_num"].astype(np.int8)
test["shop_id"] = test.shop_id.astype(np.int8)
test["item_id"] = test.item_id.astype(np.int16)
## 테스트셋 정수타입으로 수정해주기.


ts = time.time()

matrix = pd.concat([matrix, test.drop(["ID"],axis = 1)], ignore_index=True, sort=False, keys=cols)
matrix.fillna( 0, inplace = True )
time.time() - ts
## 테스트 셋과 matrix 합쳐주기


ts = time.time()
matrix = pd.merge( matrix, shops, on = ["shop_id"], how = "left" )
matrix = pd.merge(matrix, items, on = ["item_id"], how = "left")
matrix = pd.merge( matrix, cats, on = ["item_category_id"], how = "left" )
matrix["shop_city"] = matrix["shop_city"].astype(np.int8)
matrix["shop_category"] = matrix["shop_category"].astype(np.int8)
matrix["item_category_id"] = matrix["item_category_id"].astype(np.int8)
matrix["subtype_code"] = matrix["subtype_code"].astype(np.int8)
matrix["name2"] = matrix["name2"].astype(np.int8)
matrix["name3"] = matrix["name3"].astype(np.int16)
matrix["type_code"] = matrix["type_code"].astype(np.int8)
time.time() - ts

## matrix로 결합시켜줌

Feature engineering

# Define a lag feature function
def lag_feature( df,lags, cols ):
    for col in cols:
        print(col)
        tmp = df[["date_block_num", "shop_id","item_id",col ]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag_"+str(i)]
            shifted.date_block_num = shifted.date_block_num + i
            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df
## lag feature를 추가해주는 함수를 선언해줌





```


