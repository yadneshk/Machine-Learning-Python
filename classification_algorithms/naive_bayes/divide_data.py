from sklearn import datasets
import itertools
iris=datasets.load_iris()


def divide():

        label_train=[iris['target'][num:num+25] for num in range(0,len(iris['target']),50)]
        label_train=list(itertools.chain.from_iterable(label_train))



        feature_train=[iris['data'][num:num+25] for num in range(0,len(iris['data']),50)]
        feature_train= list(itertools.chain.from_iterable(feature_train))
        feature_train=[list(x) for x in feature_train]



        label_test=[iris['target'][num:num+25] for num in range(25,len(iris['target']),50)]
        label_test=list(itertools.chain.from_iterable(label_test))



        feature_test=[iris['data'][num:num+25] for num in range(25,len(iris['data']),50)]
        feature_test= list(itertools.chain.from_iterable(feature_test))
        feature_test=[list(x) for x in feature_test]

        return feature_train,label_train,feature_test,label_test


