from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix


import pandas as pd


def tfidf_process(data):
    vec = TfidfTransformer()
    vec = vec.fit(data)
    return vec


def bigram_process(data):
    vec = CountVectorizer(ngram_range=(1,2))
    vec = vec.fit(data)
    return vec


def stochastic_descent(Xtrain, Ytrain, Xtest):
    clf = SGDClassifier(loss="hinge", penalty="l1", max_iter=50)
    clf.fit(Xtrain, Ytrain)
    Ytest = clf.predict(Xtest)
    return Ytest


def linear_svm(Xtrain, Ytrain, Xtest):
    rf = LinearSVC()
    rf.fit(Xtrain, Ytrain)
    Y_test = rf.predict(Xtest)
    return Y_test


def accuracy(Ytrain, Ytest):
    acc = 0
    for i in range(len(Ytrain)):
        acc = acc + (Ytrain[i] - Ytest[i])**2
    return 100.0 - (acc/len(Ytrain))*100.0


if __name__ == "__main__":
    colnames = ['S.No', 'Text',  'Label']
    train = pd.read_csv('TrainingSet.csv', names=colnames)
    train = train[1:]
    train = train.sample(frac=1).reset_index(drop=True)
    bigram_vec = bigram_process(train['Text'])
    train_bi = bigram_vec.transform(train['Text'])

    test = pd.read_csv('TestSet.csv', names=colnames)
    test = test[1:]
    test = test.sample(frac=1).reset_index(drop=True)
    test_bi = bigram_vec.transform(test['Text'])

    tfidf_vec = tfidf_process(train_bi)
    train_tf = tfidf_vec.transform(train_bi)
    test_tf = tfidf_vec.transform(test_bi)

    #test_Y = stochastic_descent(train_tf, train['Label'], test_tf)
    #print(accuracy(test_Y, list(test['Label'])))

    print("************************")

    test_Y = linear_svm(train_tf, train['Label'], test_tf)
    print("The accuracy over the test data set is ")
    print(accuracy(test_Y, list(test['Label'])))
    print("The confusion matrix is as follows")
    print(confusion_matrix(list(test['Label']),test_Y))
    tn, fp, fn, tp = confusion_matrix(list(test['Label']),test_Y).ravel()
    print(tn, fp, fn, tp)



