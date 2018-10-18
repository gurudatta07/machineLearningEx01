from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

#knn = KNeighborsClassifier(n_neighbors=5)
#scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
#print(scores)
#print("knn mean score :"+str(scores.mean()))

k_range = range(1,45)
k_scores =[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

#print(k_scores)
plt.plot(k_range, k_scores)
plt.xlabel('K value for KNN Algorithm')
plt.ylabel('Mean Accurcy score')
plt.show()
