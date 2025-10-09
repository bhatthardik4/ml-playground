import pandas as pd

df = pd.read_csv('/Users/bhattbruh/Desktop/ml-playground/data/carprices.csv')
df

dummies = pd.get_dummies(df['Car Model'])
dummies

merged = pd.concat([df, dummies], axis='columns')
merged

final = merged.drop(['Car Model','Mercedez Benz C class'], axis='columns')
final

X = final.drop('Sell Price($)', axis='columns')
X

y = final['Sell Price($)']
y

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X, y)

model.score(X, y)

model.predict([[45000, 4, 0,0]])

model.predict([[86000, 7, 0,1]])


