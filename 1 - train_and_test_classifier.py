import numpy as np
import os
import pickle

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


def plot_confusion_matrix(cm, label_names, title='Confusion matrix', cmap=plt.cm.Blues):
    """This function create a confusion matrix from the data
    Keyword arguments:
        cm -- confusion matrix
        label_names -- array with name of each class
        title -- title for the plot
        cmap -- color to the confusion matrix
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(label_names)))
    plt.xticks(tick_marks, np.unique(label_names), rotation=45)
    plt.yticks(tick_marks, np.unique(label_names))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Load data from the dictionary created
pickle_file = 'notMNIST.pickle'
train_save_folder = 'train_data'
root = os.getcwd()

try:
    f = open(os.path.join(root, pickle_file), 'rb')
    full_data = pickle.load(f,  encoding='latin1')
except Exception as ex:
    print('Unable to load data ', pickle_file, ':', ex)
    raise

# convert train data set to matrix 2D in order to be passed to the classifier.
# original dimension: (11230, 28, 28) --> (11230, 784)
train_vector = np.reshape(full_data['train_dataset'],
                          (full_data['train_dataset'].shape[0], full_data['train_dataset'].shape[1]**2))
# convert to 1D dimension (it's required by the classifier): (11230,1) --> (11230,)
train_label = np.ravel(full_data['train_labels'])


# TRAIN CLASSIFIER MULTI-CLASS
#  set configuration parameters
logistic = LogisticRegression(C=0.01, max_iter=1000, multi_class='multinomial', solver='lbfgs')
# train a logistic classifier
logistic.fit(train_vector, train_label)
# print the score achieved in the train data
print("score in training: {:.2f}[%]".format(logistic.score(train_vector, train_label)*100))

# TEST CLASSIFIER
# convert validation data to convention followed by train data
valid_vector = np.reshape(full_data['valid_dataset'],
                          (full_data['valid_dataset'].shape[0], full_data['valid_dataset'].shape[1] ** 2))
# the same as train label
valid_label = np.ravel(full_data['valid_labels'])
# make predictions
predictions = logistic.predict(valid_vector)
prob_predictions = logistic.predict_proba(valid_vector)
print("score in training: {:.2f}[%]".format(logistic.score(valid_vector, valid_label)*100))
print("accuracy achieved in validation set: {:.2f}[%]".format(accuracy_score(valid_label, predictions)*100))
# plot confusion matrix
cm = confusion_matrix(valid_label, predictions)
print(cm)
plot_confusion_matrix(cm, label_names=np.unique(valid_label))
plt.waitforbuttonpress()
