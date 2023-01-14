># 시계열 데이터
- 일정 시간 간격으로 한 줄로 배열된 데이터를 의미한다고 할 수 있다.

## 시계열의 구성요소 


1. 추세(trend)
    추세는 장기간 데이터의 일반적인 경향을 보여준다.
    부드럽고 일반적인, 장기적인 경향으로 전체적인 추세는 상향, 하향, 혹은 안정적이어야 한다.
    짧은 구간에서는 다른 변동을 보여줄 수 있다.
    인구, 농업 생산, 제조 품목, 출생 및 사망자 수, 산업 또는 공장 수, 학교 또는 대학 수는 일종의 추세를 보여준다.

2. 순환(cycle variation)
    1년 이상 지속되는 시계열의 변동을 순환이라고 한다.
    이 변동은 1년 이상의 주기를 가지며 Business Cycle이라고 불리기도 한다.

3. 계절성(seasonal variation)
    1년 미만의 기간에 걸쳐 규칙적이고 주기적으로 나타나는 변동이다. 이러한 변동은 자연의 힘이나 사람이 만든 관습으로 인해 시작된다. 다양한 계절 또는 기후 조건은 계절 변화에 중요한 역할을 한다.
    농작물 생산량은 계절에 따라 달라지고 여름에는 냉방기기 판매량이 높아지고   겨울에는 냉방기기 판매량이 낮아지는 특징을 보이는 것을 계절성이라고 한다.
4. 불규칙 변동요인(random or irregular movements)
이 변동은 예측할 수 없고 제어할 수 없고 불규칙하다.

## 시계열 데이터의 특성
1. 시간 순차성
 - 처음부터 끝까지 일정한 시간간격으로 측정된 년 월 일 시간등의 특성을말한다.

2. 지연값(Lag)
- 관측값의 시간 차이로 발생되며 현재 관측값들은 이전 관측값들로 표현된다.

**데이터가 자기상관성을 띌 때 더 유용하다**

자기상관성이란?

- t 시점의 데이터 값이 (t-1,t-2,t-3.....t-n)시점의 데이터 값과 유의미한 상관관게를 가질 때, 자기상관을 띈다고 정의할 수 있다.


## Lag 데이터 생성 파이썬 구현

```python
ex)아래 코드는 예시임.
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
temps = DataFrame(series.values)
dataframe = concat([temps.shift(1), temps], axis=1)
# shift가 1이면 t-1  2면 t-2 3....N T-N으로 표현 가능
dataframe.columns = ['t-1', 't+1']
print(dataframe.head(5))

```
## 슬라이딩 윈도우 기법 파이썬 구현
```python

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
temps = DataFrame(series.values)
dataframe = concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
dataframe.columns = ['t-3', 't-2', 't-1', 't+1']
```


## 시계열 데이터 참고 자료
## 시계열 데이터 feature engineering
- Creating the correct input dataset to feed the ML algorithm: In this case, the purpose of feature engineering in time series forecasting is to create input features from historical row data and shape the dataset as a supervised learning problem.
  
   - ML 알고리즘에 공급할 올바른 입력 데이터 세트 생성: 이 경우 시계열 예측에서 기능 엔지니어링의 목적은 과거 행 데이터에서 입력 기능을 생성하고 데이터 세트를 지도 학습 문제로 형성하는 것입니다.



- Increasing the performance of ML models: The second most important goal of feature engineering is about generating valid relationships between input features and the output feature or target variable to be predicted. In this way, the performance of ML models can be improved.
   - ML 모델의 성능 향상: 기능 엔지니어링의 두 번째로 중요한 목표는 입력 기능과 예측할 출력 기능 또는 대상 변수 간의 유효한 관계를 생성하는 것입니다. 이러한 방식으로 ML 모델의 성능을 향상시킬 수 있습니다.

 [참고 자료](https://medium.com/data-science-at-microsoft/introduction-to-feature-engineering-for-time-series-forecasting-620aa55fcab0)


># Log scale
## Log scaling이란?
- EDA 과정을 진행하다보면 데이터가 정규분포 형태가 아닌 경우가 있다 이 경우, 그대로 모델에 데이터를 넣고 예측을 하게되면 모델의 성능이 떨어질 수 있다.

- 이를 개선해주기 위해 데이터에 로그화를 적용해주는 정규화 방법중 하나이다.


## 왜도(Skewness)와 첨도(Kurtosis)
- 왜도는 확률분포의 확률분포가 비대칭성을 띄는 것을 말한다. a=0이면 정규분포, a<0이면 오른쪽으로 치우침, a>0이면 왼쪽으로 치우침을 의미한다.

- 첨도는 확률분포의 뾰족한 정도를 말한다.

    영어로는 Kurtosis라고 하며 a=3이면 정규분포, a<3이면 정규분포보다 완만함, a>3이면 정규분포 보다 뾰족함을 의미한다.