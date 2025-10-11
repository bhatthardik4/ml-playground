import pandas as pd

df = pd.read_csv('data/titanic.csv')
df.head()

df.drop(['PassengerId', 'Name','SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df.head()

inputs = df.drop('Survived', axis=1)
target = df['Survived']
inputs.Sex = inputs.Sex.map({'male':1, 'female':2})
                             
inputs.Age[:10]
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head(10)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)
len(X_train)
len(X_test)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
model.score(X_test, y_test)