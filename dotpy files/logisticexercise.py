import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data/HR_comma_sep.csv')
df.head()

#EDA
left =df[df.left==1]
left.shape

retained = df[df.left==0]
retained.shape

df.groupby('left').mean(numeric_only=True)

pd.crosstab(df.salary, df.left).plot(kind='bar')
plt.show()

pd.crosstab(df.Department, df.left).plot(kind='bar')
plt.show()

subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
subdf.head()

salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")

df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')

df_with_dummies.head()

df_with_dummies.drop('salary', axis='columns', inplace=True)
df_with_dummies.head()  


X = df_with_dummies
X.head()

y = df.left

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)

model.predict(X_test)

model.score(X_test, y_test)