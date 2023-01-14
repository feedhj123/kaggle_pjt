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

