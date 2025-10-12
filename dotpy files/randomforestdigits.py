import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()

dir(digits)

import matplotlib.pyplot as plt
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])
    plt.show()

df = pd.DataFrame(digits.data)
df.head()
df['target'] = digits.target
df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis='columns'), df.target, test_size=0.2)
len(X_train)
len(X_test)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train) 
model.score(X_test, y_test)

y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
import seaborn as sns
plt.figure(figsize = (10,7))    
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()