from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score

def loadDataFromTxt(filename):
    data = np.genfromtxt(filename, delimiter=' ')
    return data

def loadDataFromCSV(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)[:,1:]
    return data

def main():
    q1()
    q2()
    q3()
    q4()
    q5()

def q1():
    trainingData = loadDataFromTxt('data/D2z.txt')
    print(trainingData.shape)
    gridRange = np.linspace(-2,2,num=41)
    gridPoints = []
    for x in gridRange:
        for y in gridRange:
            p = (x,y)
            gridPoints.append(p)
    gridPoints = np.array(gridPoints)

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(trainingData[:,:2], trainingData[:,2])
    gridPredictions = classifier.predict(gridPoints)

    plt.scatter(gridPoints[:,0], gridPoints[:,1], c=gridPredictions, cmap='copper', s=10)
    plt.scatter(trainingData[:,0], trainingData[:,1], c=trainingData[:,2], cmap='winter', marker='+', s=60)
    plt.show()

def q2():
    data = loadDataFromCSV('data/emails.csv')
    X = data[:,:-1]
    y = data[:,-1]
    kf = KFold(n_splits=5, shuffle=False)
    classifier = KNeighborsClassifier(n_neighbors=1)

    for i, (train, test) in enumerate(kf.split(X, y)):
        print('Fold: ', (i+1))
        classifier.fit(X[train], y[train])
        predictions = classifier.predict(X[test])
        print('Accuracy: ', accuracy_score(y[test], predictions))
        print('Precision: ', precision_score(y[test], predictions))
        print('Recall: ', recall_score(y[test], predictions))

    
def q3():
    data = loadDataFromCSV('data/emails.csv')
    X = data[:,:-1]
    y = data[:,-1]
    kf = KFold(n_splits=5, shuffle=False)

    for i, (train, test) in enumerate(kf.split(X, y)):
        print('Fold: ', (i+1))
        theta, binaryPredictions, predictions = logistic_regression(X[train], y[train], X[test])
        print('Accuracy: ', accuracy_score(y[test], binaryPredictions))
        print('Precision: ', precision_score(y[test], binaryPredictions))
        print('Recall: ', recall_score(y[test], binaryPredictions))

def q4():
    data = loadDataFromCSV('data/emails.csv')
    X = data[:,:-1]
    y = data[:,-1]
    kf = KFold(n_splits=5, shuffle=False)
    accuracyScores = []
    for i in [1,3,5,7,10]:
        classifier = KNeighborsClassifier(n_neighbors=i)
        print('k =', i)
        avg_acc = 0
        for i, (train, test) in enumerate(kf.split(X, y)):
            classifier.fit(X[train], y[train])
            predictions = classifier.predict(X[test])
            avg_acc += accuracy_score(y[test], predictions)
        avg_acc = avg_acc/5
        accuracyScores.append(avg_acc)
        print('Avg accuracy ', avg_acc)
    plt.plot([1,3,5,7,10], accuracyScores)
    plt.show()






def logistic_regression(X_train,y_train, X_test):
    theta = np.zeros(X_train.shape[1])
    learningRate = 0.001
    binaryPrediction = np.array([])
    for i in range(100):
        theta = theta - (learningRate * gradient(theta, X_train, y_train))
        prediction = np.dot(X_test, theta)
        binaryPrediction = np.where(prediction > 0, 1, 0)

    return theta, binaryPrediction, prediction


def sigmoid(x):
    return (1/(1+np.exp(-x)))

def weightedSum(x,theta):
    return np.dot(x,theta)

def costFunction(theta,X,y):
    cost = -1/x.shape[0] * np.sum(y*np.log(sigmoid(weightedSum(X,theta))) + (1-y)*np.log(1-sigmoid(weightedSum(X,theta))))
    return cost

def gradient(theta,X,y):
    return (1/X.shape[0])*np.dot(X.T, (sigmoid(weightedSum(X, theta)) - y))

def q5():
    data = loadDataFromCSV('data/emails.csv')
    X_train = data[:4000,:-1]
    y_train = data[:4000,-1]
    X_test = data[4000:,:-1]
    y_test = data[4000:,-1]

    theta, binaryPredictions, predictions = logistic_regression(X_train, y_train, X_test) 
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    print(roc_auc_score(y_test, predictions))

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    classifierPredictions = classifier.predict(X_test)
    fpr2, tpr2, thresholds2 = roc_curve(y_test, classifierPredictions)
    print(classifierPredictions.shape)
    print(roc_auc_score(y_test, classifierPredictions))


    plt.plot(fpr, tpr)
    plt.plot(fpr2, tpr2)
    plt.show()



if __name__ == '__main__':
    main()