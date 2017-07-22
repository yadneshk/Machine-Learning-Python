import random
from scipy.spatial import distance


class ScrappyKNN:
    def euc(self,a,b):
        return distance.euclidean(a,b)

    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train

    def predict(self,X_test):
        self.X_test=X_test
        pred=[]
        for row in X_test:
            label=self.closest(row)
            pred.append(label)
        return pred

    def closest(self,row):
        best_dist=self.euc(row,self.X_train[0])
        best_index=0
        for i in range(1,len(X_train)):
            dist=self.euc(row,X_train[i])
            if dist < best_dist:
                best_dist=dist
                best_index=i
        return  y_train[best_index]



from sklearn import datasets
iris=datasets.load_iris()

X=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)

#from sklearn import tree
#clf=tree.DecisionTreeClassifier()

#from sklearn.neighbors import KNeighborsClassifier
clf=ScrappyKNN()

clf.fit(X_train,y_train)
predictions=clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
#print(predictions)