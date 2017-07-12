#0=bumpy 1=smooth
from sklearn import tree
features=[[140,0],[150,0],[135,1],[155,1]]
labels=['orange','orange','apple','apple']
clf=tree.DecisionTreeClassifier()
clf.fit(features,labels)
print(clf.predict([[150,1]]))