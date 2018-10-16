from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


iris = load_iris()

X = iris.data
y = iris.target

logistic = LogisticRegression()
logistic.fit(X,y)
prediction_lr = logistic.predict([[2,4,3,1],[4,6,5,3]])
print(iris.target_names[prediction_lr[0]]+","+iris.target_names[prediction_lr[1]])

