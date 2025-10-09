import pandas as pd
df = pd.read_csv('/Users/bhattbruh/Desktop/ml-playground/data/homeprices (1).csv')

dummies = pd.get_dummies(df.town)
dummies

merged = pd.concat([df, dummies], axis='columns')
merged

final = merged.drop(['town'], axis='columns')
final

final = final.drop('west windsor', axis='columns')
final

X = final.drop('price', axis='columns')
X

Y = final.price

from sklearn.linear_model import LinearRegression
model = LinearRegression()


model.fit(X, Y)

model.predict(X)

model.score(X, y)

model.predict([[34000, 0, 0]])

model.predict([[28000, 0, 1]])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dfle = df
dfle.town = le.fit_transform(dfle.town)
dfle

X = dfle[['town', 'area']].values
X

y = dfle.price.values
y

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder='passthrough')

X = ct.fit_transform(X)
X

X = X[:, 1:]
X

model.fit(X, y)


model.predict([[1, 0, 3400]])

model.predict([[0, 1, 2800]])
model.score(X, y)