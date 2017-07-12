import random
class ScrappyKNN:
    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train

    def predict(self,X_test):
        self.X_test=X_test
        pred=[]
        for x in X_test:
            label=random.choice(self.y_train)
            pred.append(label)

        return pred


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
print(X_test)
print(y_test)
#print(predictions)