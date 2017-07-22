import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris=load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[1])
#for i in range(len(iris.target)):
#    print("Example %d: label %s ,features %s"%(i,iris.target[i],iris.data[i]))

mytesteg=[30,20,50,67,89,2,116]
training_target=np.delete(iris.target,mytesteg)
training_features=np.delete(iris.data,mytesteg,axis=0)

testing_target=iris.target[mytesteg]
testing_data=iris.data[mytesteg]

clf=tree.DecisionTreeClassifier()
clf.fit(training_features,training_target)
print(clf.predict(testing_data))
print(testing_target)

