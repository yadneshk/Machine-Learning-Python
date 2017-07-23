from sklearn.naive_bayes import GaussianNB
from divide_data import divide
from matplotlib.pyplot import plt
from sklearn.metrics import accuracy_score
feature_train,label_train,feature_test,label_test=divide()
clf=GaussianNB()
clf.fit(feature_train,label_train)
predicted_labels=clf.predict(feature_test)
print(accuracy_score(label_test,predicted_labels))
result=clf.predict_proba([[5.3,3.8,1.6,3.2]])
print(["%.3f" % float(x) for x in result[0]])
