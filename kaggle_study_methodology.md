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

## 시계열 데이터 정상성(Stationarity)
- 시계열 자료를 분석하는 통계 방법에서 빠질 수 없는 개념이다. 
- 정상성이란 늘 한결같은 성질을 의미하며, 시계열 데이터에서의 정상성이란 과거와 현재와 미래 모두 안정적이고 일정한 분포를 가진것을 말한다.
- 정상성을 유지하려면 계절성이나 추세를 띄면 안된다. 즉, 시간에 무관하게 과거와 현재와 미래의 분포가 동일해야함을 의미한다.
- 일반적인 시계열 데이터는 정상성을 제대로 갖추지 못한 경우가 많다.

떄문에, 이를 정상성을 띄게끔 변환해주기 위해서 차분과, 로그변환을 해준다.

### 차분(Differencing)
- 차분은 t시점과 t-1시점의 값의 차이를 구하는 것을 의미한다.

- 차분을 수식을 통해 정의하면 다음과 같습니다.

$$Δyt=yt−yt−1$$

첫 번째 관측값에 대한 차분 Δy1을 구할 수 없기 때문에, 차분값들은 T−1개의 값만 가지게 됩니다.

- 2차 차분
  - 가끔 차분을 해도 시계열의 정상성이 만족되지 않는 경우도 있습니다. 그럴 경우, 정상성을 나타내는 시계열을 얻기 위해 다음과 같이 한 번 더 차분을 구하는 작업이 필요할 수도 있습니다.

$$Δyt−Δyt−1=(yt−yt−1)−(yt−1−yt−2)=yt−2yt−1+yt−2$$

- 계절성 차분
  - 계절성 차분은 관측치와, 같은 계절의 이전 관측값과의 차이를 말합니다. 따라서 다음과 같이 정의가 됩니다.

   $$yt−yt−m$$
  - 여기서 m은 주기에 해당합니다. 즉 m=12이고 월별로 집계된 값이라면, 올해 1월의 값 - 작년 1월의 값, 올해 2월의 값 - 작년 2월의 값, … 이 되겠죠.

### 로그변환
- 시계열 값에 로그를 취한것

$$yt→log(yt)$$
  -  로그 변환은 특히 값의 변동 자체가 큰 경우 (= 분산이 큰 경우) 고려할 수 있는 방법입니다. 또한, GDP와 같은 많은 경제 시게열 자료가 근사적으로 지수적인 성장을 나타내고 있는 경우가 많기 때문에 이런 시계열 자료에 로그를 취해 선형적인 값으로 바꿔주는 효과 또한 있습니다.
  -  그러나 로그 변환을 통해 어떤 시계열 자료가 선형적인 추세를 보인다면
  시간의 경과에 따라 평균이 일정하지 않고 증가한다는 뜻이기 때문에 이 또한 정상성을 해친다는 문제가 있습니다.

  - 위 문제를 해결하기위해서 로그변환과 차분을 동시에 취해줍니다.


### 로그변환+차분
- 로그 변환을 통해서 먼저 분산을 안정화시키고, 지수적인 값을 가지는 시계열을 선형적으로 바꿔준 다음,

- 로그 변환된 시계열 값에 차분을 취해 정상성을 만족시키도록 만드는 것입니다.

- 로그의 차분값은 증가율의 근사값이라는 것이 알려져있다  
로그의 차분값을 다음과 같이 정의하면 

$$Δlog(yt)=logyt−logyt−1$$

이 값을 yt−1에서 yt로의 증가율로 근사할 수 있습니다.

$$Δlog(yt)=logyt−logyt−1≃yt−yt−1yt−1$$


### 정상성을 검증하는 2가지 방법

1. 그래프를 통한 검증 

- 그래프에 지속적인 상승 또는 하락 추세가 없어야 하고
- 과거의 변동폭과 현재의 변동폭이 같아야 하며
- 계절성도 없어야 합니다.
- 어떤 시점에서 평균을 크게 웃돌거나 내려갔더라도 평균으로 회귀하려는 특성이 눈에 띄어야함.

- 더 직관적을 ACF(자기상관함수)를 통해 알아보는 방법도 있다.
$$rk=∑Tt=k+1(yt−y¯)(yt−k−y¯)∑Tt(yt−y¯)2$$
여기서 , T는 시계열 데이터의 행수를 의미한다.

- ACF 그래프에서 가로축은 시차 (lag), 세로축은 ACF의 값입니다. ACF 그래프에서 그래프가 허용범위 안에 들어오면 이는 정상성을 띈다고 볼 수 있습니다.(3번 그래프)
![ACF그래프](https://assaeunji.github.io/images/stationarity-quiz.png)

2. 가설 검정으로 정상성파악하기
- 귀무 가설을 기각해야 (p-value가 작아야)정상성을 띈다고 볼 수 있다.
- **단위근 검정**이라고 부를수 있으며, KPSS,Dicky-Fuller,ADF등의 방식이 있으며 이 중 가장 보편적으로 활용되는 검정방식은 Dick-Fuller 검정 방식이다.

Ex) AR(1)모형에서의  Dicky Fuller Test
$$yt=β0+β1yt−1+et$$
여기서의 귀무 가설과 대립 가설은 
$$H0=β1=1$$ 
vs. 
$$H1:β1<1$$
로 정의가 되는데요. 귀무 가설을 만족한다면 β1=1이고, 이 때의 모형은  
$$yt=β0+yt−1+et$$
가 됩니다.

이런 모형이 어떻게 생겼는지 아래 두 모형으로 간단하게 시뮬레이션을 해보겠습니다.

모형 1: 
$$β0=3,β1=1→yt=3+yt−1+et$$

모형 2: 
$$β0=3,β1=0.1→zt=3+0.1zt−1+et$$

```python
import numpy as np
import matplotlib.pyplot as plt
init = np.random.normal(size=1, loc = 3)
e = np.random.normal(size=1000, scale = 3)

y = np.zeros(1000)
z = np.zeros(1000)
y[0] = init
z[0] = init

for t in range(1,1000):
    y[t] = 3 + 1 * y[t-1] + e[t] # beta1 = 1 단위근 모형
    z[t] = 3 + 0.1*z[t-1] + e[t] # beta1 = 0.1

fig,ax = plt.subplots(1,2)
ax[0].plot(y)
ax[0].set_title("Non-Stationary")
ax[1].plot(z)
ax[1].set_title("Stationary")
plt.show() 
```
![예시](https://assaeunji.github.io/images/stationarity-sim.png)
이처럼 β1=1이면 단위근을 가졌다고 하고, 이 단위근을 가진 AR 모형 yt 은 정상성을 띠지 않습니다. 따라서 귀무 가설 β1=1을 기각해야 정상성을 띤다 말할 수 있습니다.

```python
# ACF그래프 그리기 예시
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

fig,ax = plt.subplots(4,2, figsize=(12,16))
# ACF 그래프들
plot_acf(ss['Close'],ax=ax[0,1])
plot_acf(ss['log_close'],ax=ax[1,1])
plot_acf(ss['diff_close'][1:],ax=ax[2,1])
plot_acf(ss['logdiff_close'][1:],ax=ax[3,1])

plt.show()
```

```python
## ADF방식 재현
from statsmodels.tsa.stattools import adfuller

def print_adfuller (x):
    result = adfuller(x)
    print(f'ADF Statistic: {result[0]:.3f}')
    print(f'p-value: {result[1]:.3f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
```







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

> # 정규표현식
## 정규표현식이란?
- /regex?/ -> regular expression의 약자이다.
- 텍스트에서 우리가 원하는 특정한 패턴을 찾을때 활용 가능하다.

## 메타문자
- 메타 문자란 원래 그 문자가 가진 뜻이 아닌 특별한 용도로 사용하는 문자를 말한다.
- [. ^ $ * + ? { } [ ] \ | ( )] -< 정규표현식에서 사용되는 메타문자.


## 정규표현식의 4가지 분류
1. Groups and ranges
- | : 또는
- () : 하나의 그룹으로 묶어줌
- [] : 문자셋 가로안의 어떤 문자든
- [^] : 부정 문자셋, 괄호안의 어떤 문자가 아닐때(제외하고 다른걸 찾고싶을때 활용)
- (?:) 찾지만 기억하지는 않음

2. Quantifiers
- ? : 없거나 있거나
- *: 없거나 있거나 많거나
- +: 하나 또는 많거나
- {n}: n번반복 {n,s}-<이런식으로 최소,최대 지정가능 혹은 {n,} 이런식으로 최소만도 따로 지정가능
  
3. Boundary-type
- \b: 단어의 경계를 나타냄.
 앞에 붙으면 단어 앞에 있는것만 뒤에 붙으면 단어뒤에 있는것만 나타냄

- \B: 단어 경계가아님 (\b와 반대로 활용함)
- ^: 문장의 시작
- $: 문장의 끝

4. Character classes
- \: 특수문자가 아닌 문자 ->정규표현식에서 활용되는 문자 자체를 찾고싶을때 \.이런식으로 활용해준다.
- \d: 숫자 저분 찾기
- \D: 숫자아닌 전부
- \w: 모든 문자
- \W: 문자가아닌 것
- \s: 스페이스 공백
- \s: 스페이스 공백이 아닌것을 다 찾기 


## 파이썬에서 정규표현식
```python
import re
p = re.compile('ab*')
# re.compile을 사용하여 정규표현식을 컴파일한다.
```

- ###  정규식을 이용한 문자열 검색
|  Method |목적   |
|---|---|
| match()  |   문자열의 처음부터 정규식과 매치되는지 조사한다.|
|  search() |  문자열 전체를 검색하여 정규식과 매치되는지 조사한다. |
| findall()  |정규식과 매치되는 모든 문자열(substring)을 리스트로 리턴한다.   |
|finditer()   |정규식과 매치되는 모든 문자열(substring)을 반복 가능한 객체로 리턴한다.   |


1. match()

 match 메서드는 문자열의 처음부터 정규식과 매치되는지 조사한다. 위 패턴에 match 메서드를 수행해 보자.
```python
>>> m = p.match("python")
>>> print(m)
<re.Match object; span=(0, 6), match='python'>

>>> m = p.match("3 python")
>>> print(m)
None
"3 python" 문자열은 처음에 나오는 문자 3이 정규식 [a-z]+에 부합되지 않으므로 None을 돌려준다.
```
2. search()
- search는 match와 다르게 문자열 전체를 검색한다.
```python
>>> m = p.search("python")
>>> print(m)
<re.Match object; span=(0, 6), match='python'>

>>> m = p.search("3 python")
>>> print(m)
<re.Match object; span=(2, 8), match='python'>

# "3 python" 문자열의 첫 번째 문자는 "3"이지만 search는 문자열의 처음부터 검색하는 것이 아니라 문자열 전체를 검색하기 때문에 "3 " 이후의 "python" 문자열과 매치된다.
```

3. findall()
```pyhton
>>> result = p.findall("life is too short")
>>> print(result)
['life', 'is', 'too', 'short']
findall은 패턴([a-z]+)과 매치되는 모든 값을 찾아 리스트로 리턴한다.
```
4. finditer()
```python
>>> result = p.finditer("life is too short")
>>> print(result)
<callable_iterator object at 0x01F5E390>
>>> for r in result: print(r)
...
<re.Match object; span=(0, 4), match='life'>
<re.Match object; span=(5, 7), match='is'>
<re.Match object; span=(8, 11), match='too'>
<re.Match object; span=(12, 17), match='short'>

# finditer는 findall과 동일하지만 그 결과로 반복 가능한 객체(iterator object)를 리턴한다. 그리고 반복 가능한 객체가 포함하는 각각의 요소는 match 객체이다.
```

- ## Match객체의 메서드

|  Method |목적   |
|---|---|
| group()  |   매치된 문자열을 리턴한다.|
|  start() |  매치된 문자열의 시작 위치를 리턴한다. |
| end()  |매치된 문자열의 끝 위치를 리턴한다.   |
|span()   |매치된 문자열의 (시작, 끝)에 해당하는 튜플을 리턴한다.  |


- ## 컴파일 옵션
정규식을 컴파일할 때 다음 옵션을 사용할 수 있다.

- DOTALL(S) - . 이 줄바꿈 문자를 포함하여 모든 문자와 매치할 수 있도록 한다.

- IGNORECASE(I) - 대소문자에 관계없이 매치할 수 있도록 한다.

- MULTILINE(M) - 여러줄과 매치할 수 있도록 한다. (^, $ 메타문자의 사용과 관계가 있는 옵션이다)

- VERBOSE(X) - verbose 모드를 사용할 수 있도록 한다. (정규식을 보기 편하게 만들수 있고 주석등을 사용할 수 있게된다.)
옵션을 사용할 때는 re.DOTALL처럼 전체 옵션 이름을 써도 되고 re.S처럼 약어를 써도 된다.

```python
p = re.compile('a.b', re.DOTALL)

p = re.compile('[a-z]+', re.I)

p = re.compile("^python\s\w+", re.MULTILINE)

charref = re.compile(r"""
 &[#]                # Start of a numeric entity reference
 (
     0[0-7]+         # Octal form
   | [0-9]+          # Decimal form
   | x[0-9a-fA-F]+   # Hexadecimal form
 )
 ;                   # Trailing semicolon
""", re.VERBOSE)
re.VERBOSE 옵션을 사용하면 문자열에 사용된 whitespace는 컴파일할 때 제거된다(단 [ ] 안에 사용한 whitespace는 제외). 그리고 줄 단위로 #기호를 사용하여 주석문을 작성할 수 있다.
```

## 참조
- [정규표현식 위키독스](https://wikidocs.net/4308)
- [youtube 정규표현식](https://www.youtube.com/watch?v=t3M6toIflyQ)
- [regex 배울 수 있는 site](https://regexr.com/5ml92)

