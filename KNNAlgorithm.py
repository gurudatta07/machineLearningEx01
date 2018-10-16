from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
iris = load_iris()
#print(iris.feature_names)
#print(iris.target_names)
#print(iris.target) # Target data should be in numpy array
#print(iris.data) #input data as numpy array

X = iris.data
y = iris.target


for x in range(1,2):
    knn = KNeighborsClassifier(n_neighbors=x)
    knn.fit(X,y)
    prediction = knn.predict([[2,4,3,1],[4,6,5,3]])
    print(iris.target_names[prediction[0]]+","+iris.target_names[prediction[1]])

#for x in range(1,10):

#knn = KNeighborsClassifier(n_neighbors=1)
#print(knn)
#knn.fit(X,y)
#prediction = knn.predict([[2,4,3,1]])
#print(prediction)
#print(iris.target_names[prediction])

#new_prediction = knn.predict([[2,4,3,1],[3,4,5,6]])
#print(new_prediction)
#print(iris.target_names[new_prediction[0]]+","+iris.target_names[new_prediction[0]]) 
