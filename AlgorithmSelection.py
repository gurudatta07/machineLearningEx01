# Find out which algorithm is better
# Step 1 : Split the data into train and test

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris


iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)

#print(X.shape)
#print(X_train.shape)
#print(X_test.shape)

logisticreg = LogisticRegression()
logisticreg.fit(X_train,y_train)
y_pred = logisticreg.predict(X_test)

print("Logistic accuracy : "+str(accuracy_score(y_test,y_pred)))

