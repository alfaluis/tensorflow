import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """This function create a confusion matrix from the data
    Keyword arguments:
        cm -- confusion matrix
        title -- title for the plot
        cmap -- color to the confusion matrix
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# load iris data set
iris = datasets.load_iris()
# set a seed
np.random.seed(0)
# permute index from data
indices = np.random.permutation(len(iris.target))
# shuffle data set
iris_X_train = iris.data[indices[:-15]]
iris_y_train = iris.target[indices[:-15]]
iris_X_test = iris.data[indices[-15:]]
iris_y_test = iris.target[indices[-15:]]

# Train a classifier multi-class
# set configuration parameters
logistic = LogisticRegression(C=1, multi_class='multinomial', solver='lbfgs')
# train a logistic classifier
logistic.fit(iris_X_train, iris_y_train)
# test logistic classifier returning classes labels
prediction1 = logistic.predict(iris_X_test)
# test logistic classifier returning classes probabilities
prediction2 = logistic.predict_proba(iris_X_test)

# create a confusion matrix
cm = confusion_matrix(iris_y_test, prediction1)
print(cm)
plot_confusion_matrix(cm)
plt.waitforbuttonpress()


