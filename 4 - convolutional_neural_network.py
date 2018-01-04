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
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels).astype(np.float32)
    return dataset, labels

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy_fun(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def conv_network(data, weights, biases, padding='SAME', stride=[1, 1, 1, 1]):
    """ Model (Graph) for convolutional neural network
    
    :param data: A 4D matrix. Where the first dimension contains the batch size, the second and third the height and 
                 width of image and the four correspond to the depth of the image (3 in RGB and 1 in grayscale) 
    :param weights: dictionary with all model's weight.
    :param biases:  dictionary with all model's biases.
    :param padding: String with 'SAME' or 'VALID' padding. Add or not zero padding to the image
    :param stride: Set the sliding step for each dimension. [batch, height, width, depth]. By default = [1, 2, 2, 1],
                   The stride is sliding by each image in the batch (stride[0]), steps inside the image by 2 pixels 
                   (stride[1], stride[2]) and do it in each depth of the image (stride[-1]. In the case of RGB image,
                   we operate over each color)
    :return: Result over each image in the batch for the model 
    """
    # initial image of BATCHx28x28x1. Convolution with stride [2,2]
    conv = tf.nn.conv2d(data, weights['w1'], stride, padding=padding)
    hidden = tf.nn.relu(conv + biases['b1'])
    # output image of BATCHx14x14x32. Convolution with stride [2,2]
    conv = tf.nn.conv2d(hidden, weights['w2'], stride, padding=padding)
    hidden = tf.nn.relu(conv + biases['b2'])
    # output image of BATCHx7x7x32.
    shape = hidden.get_shape().as_list()
    print('output second convolution: ' + str(shape))
    # resshape for full connected layer. [BATCH, 7*7*32] -> [BATCH, 1568]
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    print('input fc ann: ' + str(reshape.shape))
    # full connected layer
    hidden = tf.nn.relu(tf.matmul(reshape, weights['w3']) + biases['b3'])
    # output layer
    return tf.matmul(hidden, weights['w4']) + biases['b4']


graph = tf.Graph()
valid_index = np.random.randint(0, valid_dataset.shape[0], 1000)
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
num_steps = 1001
with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset[valid_index, :])
    tf_test_dataset = tf.constant(test_dataset)
    global_step = tf.Variable(initial_value=0, trainable=False)
    # Variables.
    # diagram for convnet
    # [28x28x1] -> conv(filter=[5x5],stride=2, depth=32) -> [14x14x32] (output first convolution)
    # [14x14x32] -> conv(filter=[5x5],stride=2, depth=32) -> [7x7x32] (output second convolution)
    # [7x7x32] -> [1568, 1] -> fc[1568, 128] -> [128,1] (output first fc layer)
    # [128, 1] -> fc[128, 10] -> [10, 1] (output layer - predictions)
    weight = {'w1': tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1)),
              'w2': tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1)),
              'w3': tf.Variable(tf.truncated_normal([image_size//4 * image_size//4 * depth, num_hidden], stddev=0.1)),
              'w4': tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))}
    bias = {'b1': tf.Variable(tf.zeros([depth])),
            'b2': tf.Variable(tf.constant(1.0, shape=[depth])),
            'b3': tf.Variable(tf.constant(1.0, shape=[num_hidden])),
            'b4': tf.Variable(tf.constant(1.0, shape=[num_labels]))}

    # Training computation.
    logits = conv_network(tf_train_dataset, weights=weight, biases=bias, stride=[1, 2, 2, 1])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # learning decay
    # own implementation
    decay_steps = num_steps
    decay_rate = 0.96
    learning_rate = 0.1 * decay_rate ** (global_step/decay_steps)
    # by function. In contrast to the manual implementation, the function added  a staircase which convert the result
    # to a integer
    learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=num_steps,
                                               decay_rate=decay_rate, staircase=False)
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    print(valid_dataset[valid_index, :].shape)
    valid_prediction = tf.nn.softmax(conv_network(tf_valid_dataset, weights=weight, biases=bias, stride=[1, 2, 2, 1]))
    test_prediction = tf.nn.softmax(conv_network(tf_test_dataset, weights=weight, biases=bias, stride=[1, 2, 2, 1]))


# valid_index = np.random.randint(0, valid_dataset.shape[0], 1000)
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        idx = np.random.randint(0, train_dataset.shape[0], batch_size)
        batch_data = train_dataset[idx, :]
        batch_labels = train_labels[idx, :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 50 == 0):
            print(tf.train.global_step(sess=session, global_step_tensor=global_step))
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy_fun(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy_fun(valid_prediction.eval(), valid_labels[valid_index, :]))
    print('Test accuracy: %.1f%%' % accuracy_fun(test_prediction.eval(), test_labels))


# **********************************************************************************************
# Problem 1
# Replace the strides by a max pooling operation (nn.max_pool()) of stride 2 and kernel size 2
# **********************************************************************************************

def weight_variable(shape):
    # shape has the format [Height, Width, Channels, Output]
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(value=0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w, stride=[1, 1, 1, 1], pad='SAME'):
    """
    Apply a convolutional operation to the input image 'x' using the filter 'w'. Moving across the image defined by 
    stride and adding or not padding defined by pad
    
    :param x: Input image. It must have the shape [N° samples, Heigh, Width, channels]
    :param w: Filter. Format must be [height, width, input_depth, output_depth]
    :param stride: Define the way that the filter is moving across the image and samples. 
                   Format [sample_step, height_step, width_step, channel_step]
    :param pad: Define if in the input image will be used or not zero padding. Two able values 'SAME' or 'VALID'. 
    :return: Tensor with the output image after convolution
    """

    return tf.nn.conv2d(data_format='NHWC', input=x, filter=w, strides=stride, padding=pad)


def max_pool_2x2(x):
    """ make a max pool operation with a filter size of 2x2 and stride sample_step=1, height_step=2, width_step=2 and 
    channel_step=1
    :param x: Input image. It must have the shape [N° samples, Heigh, Width, channels]
    :return: 
    """

    return tf.nn.max_pool(data_format='NHWC', value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def exp_learning_rate(start_lr, final_lr, total_step, step):
    # return start_lr * final_ln ** (step / total_step)
    return start_lr - (start_lr - final_lr) * (step / total_step)


def deep_cnn(x):
    # first layer filter size 5x5x1 and 32 output depth
    # input: 28x28x1; output: 14x14x32
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second layer filter size 5x5x32 and 32 output depth
    # input: 14x14x32; output: 7x7x32
    w_conv2 = weight_variable([5, 5, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # output image of BATCHx7x7x32.
    shape = h_pool2.get_shape().as_list()
    print('output second convolution: ' + str(shape))
    # reshape for full connected layer. [BATCH, 7*7*32] -> [BATCH, 1568]
    reshape = tf.reshape(h_pool2, [-1, shape[1] * shape[2] * shape[3]])
    print('input fc ann: ' + str(reshape.shape))

    # full connected layer
    w_fc1 = weight_variable([shape[1] * shape[2] * shape[3], 32])
    b_fc1 = bias_variable([32])
    keep_prob = tf.placeholder(tf.float32)
    h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # output layer
    w_fc2 = weight_variable([32, num_labels])
    b_fc2 = bias_variable([num_labels])
    y_out = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    print(y_out.get_shape().as_list())
    return y_out, keep_prob

# main
batch_size = 64
num_steps = 1000
valid_index = np.random.randint(0, valid_dataset.shape[0], 1000)
global_step = tf.Variable(initial_value=0, trainable=False)
x_data = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
y_data = tf.placeholder(tf.float32, shape=(None, num_labels))

# create deep learning model
logits, keep_prob = deep_cnn(x_data)
# train model
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=logits))
learning_rate = exp_learning_rate(start_lr=0.5, final_lr=0.1, step=global_step, total_step=num_steps)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
# predictions in two ways
train_prediction = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_data, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        # print(tf.train.global_step())
        idx = np.random.randint(0, train_dataset.shape[0], batch_size)
        batch_data = train_dataset[idx, :]
        batch_labels = train_labels[idx, :]
        feed_dict = {x_data: batch_data, y_data: batch_labels, keep_prob: 0.5}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 50 == 0:
            print(tf.train.global_step(sess=session, global_step_tensor=global_step))
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy_fun(predictions, batch_labels))
            validation = accuracy.eval(feed_dict={x_data: valid_dataset[valid_index, :],
                                                  y_data: valid_labels[valid_index, :],
                                                  keep_prob: 1})
            print('Validation accuracy: %.1f%%' % (validation * 100))
    test_prediction = accuracy.eval(feed_dict={x_data: test_dataset,
                                               y_data: test_labels,
                                               keep_prob: 1})
    test_2 = train_prediction.eval(feed_dict={x_data: test_dataset,
                                              y_data: test_labels,
                                              keep_prob: 1})
    print('Test accuracy: %.1f%%' % accuracy_fun(test_2, test_labels))
    print('Test accuracy: %.1f%%' % (test_prediction * 100))



