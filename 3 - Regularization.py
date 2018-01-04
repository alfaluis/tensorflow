import numpy as np
import tensorflow as tf
import os
import pickle

pickle_file = 'notMNIST.pickle'
root = os.getcwd()
with open(os.path.join(root, pickle_file), 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

"""
Reformat into a shape that's more adapted to the models we're going to train:

    data as a flat matrix,
    labels as float 1-hot encodings.
"""


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels).astype(np.float32)
    return dataset, labels

image_size = 28
num_labels = 10
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

"""
Question one: adding regularization to artificial neural network
"""

g_ann_reg = tf.Graph()
batch_size = 128
hidden_nodes = 1024
image_size = 28
beta = 0.01

with g_ann_reg.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    keep_prob = tf.placeholder(tf.float32)

    w1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_nodes]))
    b1 = tf.Variable(tf.zeros([hidden_nodes]))

    w2 = tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))
    b2 = tf.Variable(tf.zeros([num_labels]))

    logits1 = tf.matmul(tf_train_dataset, w1) + b1
    hidden = tf.nn.relu(logits1)
    logits2 = tf.matmul(hidden, w2) + b2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits2)) \
           + (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)) * beta
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(logits2)
    valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, w1) + b1), w2) + b2)
    test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, w1) + b1), w2) + b2)

"""
Question 1/2: Train and Test regularized model and also change the numbers of step to a few ones
num_step for normal training should be at least 2000.
num_step for reduced training should be leaser than 100
Obviously with the number of step reduced the model work bad  
"""
num_steps = 50

with tf.Session(graph=g_ann_reg) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        idx = np.random.randint(0, train_dataset.shape[0], batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[idx, :]
        batch_labels = train_labels[idx, :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: .75}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 5 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


"""
Question 3: adding a dropout to the neural network
"""
g_ann_drop = tf.Graph()
batch_size = 128
hidden_nodes = 1024
image_size = 28
beta = 0.01

with g_ann_drop.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    keep_prob = tf.placeholder(tf.float32)

    w1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_nodes]))
    b1 = tf.Variable(tf.zeros([hidden_nodes]))

    w2 = tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))
    b2 = tf.Variable(tf.zeros([num_labels]))

    logits1 = tf.matmul(tf_train_dataset, w1) + b1

    hidden = tf.nn.relu(logits1)

    hidden_dropout = tf.nn.dropout(hidden, keep_prob)

    logits2 = tf.matmul(hidden_dropout, w2) + b2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits2))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(logits2)
    valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, w1) + b1), w2) + b2)
    test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, w1) + b1), w2) + b2)


num_steps = 3000

with tf.Session(graph=g_ann_drop) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        idx = np.random.randint(0, train_dataset.shape[0], batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[idx, :]
        batch_labels = train_labels[idx, :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: .75}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))



"""
Question 4: creating a new neural network with a first layer with 1024 nodes and relu activation function, the second 
layer contains 512 nodes relu activation and dropout. All layer are with L2 regularization 
"""
g_ann_new = tf.Graph()
batch_size = 128
hidden_1 = 1024
hidden_2 = 1024
image_size = 28
beta = 0.01
alpha = 0.05


def ann_two_layers(data, prob, weight, bias):
    """ Create a model (graph) for an artificial neural network compose by to layers configurable, that means, the 
    number of nodes depending by weights and biases
    
    :param data: Data to evaluate in the model. 
    :param prob: probability to keep in dropout
    :param weight: dictionary with weights for the model
    :param bias: dictionary with biases for the model
    :return: Final output for the model
    """
    # first layer
    hidden1_output = tf.matmul(data, weight['w1']) + bias['b1']
    hidden1_relu = tf.nn.relu(hidden1_output)
    # second layer
    hidden2_output = tf.matmul(hidden1_relu,  weight['w2']) + bias['b2']
    hidden2_relu = tf.nn.relu(hidden2_output)
    hidden2_dropout = tf.nn.dropout(hidden2_relu, prob)
    # output
    return tf.matmul(hidden2_dropout,  weight['w3']) + bias['b3']

with g_ann_new.as_default():
    # data
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # parameters
    keep_prob = tf.placeholder(tf.float32)
    weight = {'w1': tf.Variable(tf.truncated_normal([image_size * image_size, hidden_1])),
              'w2': tf.Variable(tf.truncated_normal([hidden_1, hidden_2])),
              'w3': tf.Variable(tf.truncated_normal([hidden_2, num_labels]))}
    bias = {'b1': tf.Variable(tf.zeros([hidden_1])),
            'b2': tf.Variable(tf.zeros([hidden_2])),
            'b3': tf.Variable(tf.zeros([num_labels]))}

    # train model
    logits = ann_two_layers(data=tf_train_dataset, weight=weight, bias=bias, prob=keep_prob)
    regularization = beta * (tf.nn.l2_loss(weight['w1']) + tf.nn.l2_loss(weight['w2']) + tf.nn.l2_loss(weight['w3']))
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) + regularization
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    # predictions
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(ann_two_layers(data=tf_valid_dataset,
                                                    prob=1,
                                                    weight=weight,
                                                    bias=bias))

    test_prediction = tf.nn.softmax(ann_two_layers(data=tf_test_dataset,
                                                   prob=1,
                                                   weight=weight,
                                                   bias=bias))


num_steps = 8001

with tf.Session(graph=g_ann_new) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Generate a random minibatch.
        idx = np.random.randint(0, train_dataset.shape[0], batch_size)
        batch_data = train_dataset[idx, :]
        batch_labels = train_labels[idx, :]

        # Prepare a dictionary telling the session where to feed the minibatch.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: .75}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        # print each 500 iterations
        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
