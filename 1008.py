#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[106]:


df_train = pd.io.parsers.read_csv("titanic_train.csv")


# In[107]:


df_train.head()


# In[108]:


df_test = pd.read_csv("titanic_train.csv")


# In[109]:


df_test.info()


# In[110]:


# 불피요한 피처 값들을 삭제 (중복실행시 에러 발생 주의)
df_train = df_train.drop(['name','ticket','body','cabin','home.dest'],axis = 1)


# In[111]:


df_train.head()


# In[112]:


df_test = df_test.drop(['name','ticket','body','cabin','home.dest'], axis =1)


# In[113]:


df_test.head()


# In[114]:


print(df_train['survived'].value_counts())
# 1 : 생존자 0 : 고인자


# In[115]:


df_train['survived'].value_counts().plot.bar()


# In[116]:


# survived 피처를 기준으로 그룹을 나누어 그룹별 pclass 피처의 분포를 살펴보자
# pclass 승객 등급

print(df_train['pclass'].value_counts())


# In[117]:


ax = sns.countplot(x='pclass', hue = 'survived', data = df_train)


# In[118]:


# 변수 탐색적인 자동화 필수

from scipy import stats

# 두 집단의 피처를 비교해주며 탐색작업을 자동화하는 함수를 정의합니다.
def valid_features(df, col_name, distribution_check=True):
    
    # 두 집단 (survived=1, survived=0)의 분포 그래프를 출력합니다.
    g = sns.FacetGrid(df, col='survived')
    g.map(plt.hist, col_name, bins=30)

    # 두 집단 (survived=1, survived=0)의 표준편차를 각각 출력합니다.
    titanic_survived = df[df['survived']==1]
    titanic_survived_static = np.array(titanic_survived[col_name])
    print("data std is", '%.2f' % np.std(titanic_survived_static))
    titanic_n_survived = df[df['survived']==0]
    titanic_n_survived_static = np.array(titanic_n_survived[col_name])
    print("data std is", '%.2f' % np.std(titanic_n_survived_static))
    
     # T-test로 두 집단의 평균 차이를 검정합니다.
    tTestResult = stats.ttest_ind(titanic_survived[col_name], titanic_n_survived[col_name])
    tTestResultDiffVar = stats.ttest_ind(titanic_survived[col_name], titanic_n_survived[col_name], equal_var=False)
    print("The t-statistic and p-value assuming equal variances is %.3f and %.3f." % tTestResult)
    print("The t-statistic and p-value not assuming equal variances is %.3f and %.3f" % tTestResultDiffVar)
    
    if distribution_check:
        # Shapiro-Wilk 검정 : 분포의 정규성 정도를 검증합니다.
        print("The w-statistic and p-value in Survived %.3f and %.3f" % stats.shapiro(titanic_survived[col_name]))
        print("The w-statistic and p-value in Non-Survived %.3f and %.3f" % stats.shapiro(titanic_n_survived[col_name]))


# In[119]:


# 함수 실행 age, sibsp

valid_features(df_train[df_train['age']>0],'age',distribution_check=True)


# In[120]:


valid_features(df_train,'sibsp',distribution_check=False)


# In[121]:


# parch : 동승한 부모 또는 자녀수
# sex : 탑승자의 성별

valid_features(df_train,'sibsp',distribution_check = True)
# sibsp 동승한 형제 또는 배우자 수


# In[138]:


valid_features(df_train,'pclass',distribution_check=True)

# pclass (고객등급)          영향을 미친다.
# age (나이)                :  ?
# sibsp, parch (동승자)     :  ?
# sex (성)                   영향을 미친다. 


# In[ ]:


valid_features(df_train,'parch',distribution_check=True)


# In[ ]:


valid_features(df_train,'pclass',distribution_check=True)

# pclass (고객등급) : 영향을 미친다.
# age (나이)        : ?
# slbsp, parch      : ?
# sex (성)          : 영향을 미친다.


# In[ ]:


# 로지스틱 회귀 모델
# 기존의 회귀 분석의 예측값 Y를 0-1 사이의 값으로 제한
# ***********0.5보다 크면           0.5보다 작으면**********
# 으로 분류하는 방법, 개수 분석을 통한 피처의 영향엵 해석이 용이하다는 장점을 가진다.

## 전처리 1) 결측이 존재하는 데이터를 삭제 
##        2) 평균값, 또는 중앙값 또는 최빈값 등의 임의의 수치로 채워 넣는 방법
##        3) 데이터를 모두 분석에 활용할수 있는 장점인 반면에 수치 왜곡의 가능성
##        4) 2)번 방법을 사용하여 전처리
##        5) age의 결측값을 평균값으로 대체하다.

replace_mean = df_train[df_train['age'] > 0]['age'].mean()
df_train['age'] = df_train['age'].fillna(replace_mean)
df_test['age'] = df_test['age'].fillna(replace_mean)

#embark의 결측값을 최빈값 대체하자
embarked_mode = df_train['embarked'].value_counts().index[0]
df_train['embarked'] = df_train['embarked'].fillna(embarked_mode)
df_test['embarked'] = df_test['embarked'].fillna(embarked_mode)

# 원-핫 인코딩, 
# 통합 데이터 프레임 (whwole_df) 생성
whole_df = df_train.append(df_test)
train_idx_num = len(df_train)

# pandas 패키지를 이용해서 원-핫 인코딩 수행
whole_df_encoded = pd.get_dummies(whole_df)
df_train = whole_df_encoded[:train_idx_num]
df_test = whole_df_encoded[:train_idx_num]

df_train.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[ ]:



# 데이터를 학습 데이터와 테스트 데이터로 분리

x_train, y_train = df_train.loc[:, df_train.columns != 'survived'].values, df_train['survived'].values


# In[ ]:


x_test, y_test = df_train.loc[:, df_train.columns != 'survived'].values, df_train['survived'].values


# In[ ]:


# 로지스틱 회귀 모델 학습

lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)


# In[123]:


y_pred = lr.predict(x_test)
print(y_pred)


# In[124]:


y_pred_p = lr.predict_proba(x_test)[:,1]
print(y_pred_p)


# In[125]:


# 분류 모델 평가
# 테스트 데이터에 대한 정확도, 정밀도, 특이도, 평가 지표

print("정확도 : %.2f" % accuracy_score(y_test, y_pred))
print("정밀도 : %.2f" % precision_score(y_test, y_pred))
print("특이도 : %.2f" % recall_score(y_test, y_pred))
print("평가지표 : %2f" % f1_score(y_test, y_pred))


# In[126]:


# 로지스틱 희귀 모델과 더불어 분류 분석의 가장 대표적인 방법인
# 의사결정나무(decision Tree) 모델을 적용해 보자

# 의사결정나무 모델은 피처 단위로 조건을 분기하여 정답의 집합을 좁혀나가는 방법
# 마치 스무고개 놀이에서 정답을 찾아 나가는 과정과 유사하다.

#                         남자?
#                 예                노
#             나이 > 10?                  생존 0.73
#         예           노
#     사망                 가족
# 0.17%                 사망   생존

from sklearn.tree import DecisionTreeClassifier

# 의사 결정 나무를 학습하고, 학습한 모델로 테스트 데이터셋에 대한 예측값을 반환한다.
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)

print(y_pred)

y_pred_p = dtc.predict_proba(x_test)[:,1]
print(y_pred_p)


# In[127]:


#====================================================
# 모델 개선

# 분류 모델의 성능을 더욱 끌어올리기 위해서...
# 1) 좋은 분류 기법을 사용해야한다.
# 2) 더 많은 데이터를 사용한다.
# 3) 피처 엔지니어링 Feature Engineering
# 4) 피처 엔지니어링은 모델에 사용할 피처를 가공하는 분석 작업을 말한다.

df_test = pd.read_csv("titanic_train.csv")
df_train = pd.io.parsers.read_csv("titanic_train.csv")


# In[128]:


# 분류 모델 평가
# 테스트 데이터에 대한 정확도, 정밀도, 특이도, 평가 지표

print("정확도 :%.2f" % accuracy_score(y_test, y_pred))
print("정밀도 :%.2f" % precision_score(y_test, y_pred))
print("특이도 :%.2f" % recall_score(y_test, y_pred))
print("평가지표 :%.2f" % f1_score(y_test, y_pred))

# 회귀분석 결과와 비교
# 정확도 :0.79
# 정밀도 :0.73
# 특이도 :0.70
# 평가지표 :0.72


# In[129]:


# age의 결측값을 평균값으로 대체
age_mode = df_train['age'].value_counts().index[0]
df_train['age'] = df_train['age'].fillna(embarked_mode)
df_test['age'] = df_test['age'].fillna(embarked_mode)

# embark 결측값을 최빈값으로 대체
embarked_mode = df_train['embarked'].value_counts().index[0]
df_train['embarked'] = df_train['embarked'].fillna(embarked_mode)
df_test['embarked'] = df_test['embarked'].fillna(embarked_mode)
# 원-핫 인코딩을 위해서 통합데이터셋 whole_df를 생성한다.
whole_df = df_train.append(df_test)
train_idx_num = len(df_train)


# In[130]:


#피처 엔지니어링 & 전처리 

# cabin 피처 활용

# 결측 데이터를 'X' 대체
whole_df['cabin'] = whole_df['cabin'].fillna('X')

# cabin 피처의 첫 번째 알파벳을 추출하다.
whole_df['cabin'] = whole_df['cabin'].apply(lambda x:x[0])

# 추출한 알파벳 중에 G T 수가 너무작기 때문에 X 대체
whole_df['cabin'] = whole_df['cabin'].replace({"G":"X", "T":"X"})

ax = sns.countplot(x='cabin', hue = 'survived', data = whole_df)
plt.show()


# In[131]:


# name 피처 성 호칭 이름

whole_df.head()

name_grade = whole_df['name'].apply(lambda x : x.split(", ",1)[1].split(".")[0])
name_grade = name_grade.unique().tolist()
print(name_grade)


# In[132]:


# 호칭에 따라서 사회적 지위를 정의 (1910 기준)
grade_dict = {'A':['Rev','Col','Major','Dr','Capt','Sir'],
             'B' : ['Ms','Mme','Mrs','Dona'],
             'C' : ['Jonkheer', 'the Countess'],
             'D' : ['Mr','Don'],
             'E' : ['Master'],
             'F' : ['Miss','Mile','Lady']}


# In[133]:


# 정의한 호칭의 기준에 따라, A-F의 문자로 name 피처를 다시 정의하는 함수입니다.
def give_grade(x):
    grade = x.split(", ", 1)[1].split(".")[0]
    for key, value in grade_dict.items():
        for title in value:
            if grade == title:
                return key
    return 'G'

# 위의 함수를 적용하여 name 피처를 새롭게 정의합니다.
whole_df['name'] = whole_df['name'].apply(lambda x : give_grade(x))
print(whole_df['name'].value_counts())


# In[134]:


print(whole_df)


# In[137]:


whole_df_encoded = pd.get_dummies(whole_df)
df_train = whole_df_encoded[:train_idx_num]
df_test = whole_df_encoded[:train_idx_num:]
df_train.head()


# In[142]:


# 데이터를 학습 데이터와 테스트 데이터로 분리

x_train, y_train = df_train.loc[:, df_train.columns != 'survived'].values, df_train['survived'].values
x_test, y_test = df_train.loc[:, df_train.columns != 'survived'].values, df_train['survived'].values


# In[143]:


# 로지스틱 회귀 모델 학습

lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)


# In[140]:


# 결측 데이터를 'X' 대체
whole_df['body'] = whole_df['body'].fillna('0')


# In[141]:


df_train.head(10)


# In[ ]:




