import numpy as np
import tensorflow as tf
import os
import pickle


def load_database(train_folder, file_name):
    try:
        f = open(os.path.join(train_folder, file_name), 'rb')
        full_data = pickle.load(f,  encoding='latin1')
        train_dataset = full_data['train_dataset']
        train_labels = full_data['train_labels']
        valid_dataset = full_data['valid_dataset']
        valid_labels = full_data['valid_labels']
        del full_data  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
    except Exception as ex:
        print('Unable to load data ', file_name, ':', ex)
        raise
    return train_dataset, train_labels, valid_dataset, valid_labels


def reformat(dataset, labels):
    # it is the same as reshape(train_dataset.shape[0], image_size * image_size)
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels).astype(np.float32)
    return dataset, labels


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def train_test_gradient_descend(graph, train_dataset, train_labels, valid_dataset, index_subset, num_steps=801):
    with graph.as_default():
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(valid_dataset)

        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random values following a (truncated)
        # normal distribution. The biases get initialized to zero.
        weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        """ EXAMPLE
        as we see in the videos, the output from the linear regression pass by a softmax function and later in order to 
        compare or get the distance between the classifier's output and the target we apply cross entropy.
        The function tf.nn.softmax_cross_entropy_with_logits() do exactly that.
        Example:
        target = np.array([[1,0,0],[1,0,0]]).astype(np.float32)
        prediction = np.array([[3.0,2.6,5.4],[2.6,5.1,7.0]]).astype(np.float32)
        # should produce the same output as ... (axis=1 indicate along all columns)
        softmax_out = session.run(tf.nn.softmax(prediction))
        cross_entropy_out = -np.sum(np.multiply(target, np.log(softmax_out)), axis=1)
        loss = np.mean(cross_entropy_out)
        # this operation
        session.run(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=prediction))
        session.run(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=prediction)))
        """
        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)

    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases.
        session.run(tf.global_variables_initializer())
        print('Initialized')
        for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            _, l, predictions = session.run([optimizer, loss, train_prediction])
            if step % 100 == 0:
                print('Loss at step %d: %f' % (step, l))
                print('Training accuracy: %.1f%%' % accuracy(
                    predictions, train_labels[:train_subset, :]))
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), valid_labels))
                # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

    return weights, biases


def train_test_stochastic_gradient_descend():
    pass


def train_test_ann():
    pass


"""
Reformat into a shape that's more adapted to the models we're going to train:

    data as a flat matrix,
    labels as float 1-hot encodings.
"""

# Load data from the dictionary created
pickle_file = 'notMNIST.pickle'
train_save_folder = 'train_data3'
root = os.getcwd()
image_size = 28
num_labels = 10


train_dataset, train_labels, valid_dataset, valid_labels = load_database(train_folder=root,
                                                                         file_name=pickle_file)

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)

"""
We're first going to train a multinomial logistic regression using simple gradient descent.
TensorFlow works like this:
    First you describe the computation that you want to see performed: what the inputs, the variables, 
    and the operations look like. These get created as nodes over a computation graph. 
    This description is all contained within the block below:
    with graph.as_default():
        ...

    Then you can run the operations on this graph as many times as you want by calling session.run(), 
    providing it outputs to fetch from the graph that get returned. 
    This runtime operation is all contained in the block below:
    with tf.Session(graph=graph) as session:
        ...

Let's load all the data into TensorFlow and build the computation graph corresponding to our training:
"""


# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 30000
num_steps = 801
graph = tf.Graph()
idx = np.random.randint(0, train_dataset.shape[0], train_subset)
train_test_gradient_descend(graph, train_dataset, train_labels, valid_dataset, index_subset=idx, num_steps=801)

with graph.as_default():
    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.
    tf_train_dataset = tf.constant(train_dataset[idx, :])
    tf_train_labels = tf.constant(train_labels[idx, :])
    tf_valid_dataset = tf.constant(valid_dataset)

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    """ EXAMPLE
    as we see in the videos, the output from the linear regression pass by a softmax function and later in order to 
    compare or get the distance between the classifier's output and the target we apply cross entropy.
    The function tf.nn.softmax_cross_entropy_with_logits() do exactly that.
    Example:
    target = np.array([[1,0,0],[1,0,0]]).astype(np.float32)
    prediction = np.array([[3.0,2.6,5.4],[2.6,5.1,7.0]]).astype(np.float32)
    # should produce the same output as ... (axis=1 indicate along all columns)
    softmax_out = session.run(tf.nn.softmax(prediction))
    cross_entropy_out = -np.sum(np.multiply(target, np.log(softmax_out)), axis=1)
    
    # this operation
    session.run(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=prediction))
    """
    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)


with tf.Session(graph=graph) as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the
    # biases.
    session.run(tf.global_variables_initializer())
    print('Initialized')
    for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        _, l, predictions, p2 = session.run([optimizer, loss, train_prediction, valid_prediction])
        if step % 100 == 0:
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(
                predictions, train_labels[idx, :]))
            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph
            # dependencies.
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
    #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


"""
Let's now switch to stochastic gradient descent training instead, which is much faster.
The graph will be similar, except that instead of holding all the training data into a constant node, 
we create a Placeholder node which will be fed actual data at every call of session.run().
"""

num_steps = 5001
batch_size = 1000
graph = tf.Graph()

with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    #tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    #test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        idx = np.random.randint(0, train_dataset.shape[0], batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[idx, :]
        batch_labels = train_labels[idx, :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


"""
In the third part we are going to switch the model to an artificial neural network
"""

g_ann = tf.Graph()
batch_size = 1000
hidden_nodes = 1024
image_size = 28

with g_ann.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)

    w1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_nodes]))
    b1 = tf.Variable(tf.zeros([hidden_nodes]))

    w2 = tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))
    b2 = tf.Variable(tf.zeros([num_labels]))

    logits1 = tf.matmul(tf_train_dataset, w1) + b1

    hidden = tf.nn.relu(logits1)

    logits2 = tf.matmul(hidden, w2) + b2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits2))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(logits2)
    valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, w1) + b1), w2) + b2)


num_steps = 4001

with tf.Session(graph=g_ann) as session:
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
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    # print("Validation accuracy: %.1f%%" % accuracy(test_prediction.eval(), valid_labels))


