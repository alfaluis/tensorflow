import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        name = f.namelist()[0]
        data = tf.compat.as_str(f.read(name))
    return data


def char2id(char):
    """ This function convert a char to id.
    Example:
        first_letter = ord(a) -> 97
        d -> ord(d) -> 100 (DEC) -> ord(d) - ord(a) + 1 = 100 - 97 + 1 = 4
    :param char: a char to convert
    :return: a number that represent a char. Where a=1, b=2, ...
    """
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0


def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data.
            Example:
                text = 'the house is a big place where sleep'
                batch_size = 5
                segment = len(text) // batch_size
                cursor = [offset * segment for offset in range(batch_size)]
                batch = np.zeros(shape=[batch_size, 27])
                for b in range(batch_size):
                    batch[b, char2id(text[cursor[b]])] = 1
                    print(text[cursor[b]], )
                    cursor[b] += 1  # the real code have overflow control

                # print -> (t, 0); (s, 7); (' ', 14); (a, 21); (r, 28)
        """
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            # map letter to one hot. This process take a letter jumping by the whole text given by cursor
            # index value
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            # enable restart
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def characters(probabilities):
    """ Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]


def batches2string(batches):
    """ Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s


filename = maybe_download('text8.zip', 31344016)
text = read_data(filename)
print('Data size %d' % len(text))

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0))

batch_size = 64
num_unrollings = 10

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))

num_nodes = 64

# Parameters:
# Input gate: input, old output cell, and bias.
ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
ib = tf.Variable(tf.zeros([1, num_nodes]))
# Forget gate: input, old output, and bias.
fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
fb = tf.Variable(tf.zeros([1, num_nodes]))
# Memory cell: input, old state cell and bias.
cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
cb = tf.Variable(tf.zeros([1, num_nodes]))
# Output gate: input, old output, and bias.
ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
ob = tf.Variable(tf.zeros([1, num_nodes]))
# Variables saving state across unrollings.
saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
# Classifier weights and biases.
w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
b = tf.Variable(tf.zeros([vocabulary_size]))


def lstm_cell(i, o, state):
    """ Create a LSTM cell. See e.g.: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

    :param i: Input sample
    :param o: Old output of the cell (h_t-1)
    :param state: Old state of the cell (C_t-1)
    :return: New output cell (h_t), New state cell (C_t)
    """
    # input gate = w_xi * i + w_hi * o + b_i
    i_t = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    # forget gate = w_xf * i + w_hf * o + b_f
    f_t = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    # cell candidate = tanh(w_xc * i + w_hc * o + b_c)
    c_tau = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
    # new cell state = forget * old_cell_state + input * cell_candidate
    c_t = f_t * state + i_t * tf.tanh(c_tau)
    # output gate = sigmod(w_xo * i + w_ho * o + b_o)
    o_t = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    return o_t * tf.tanh(c_t), c_t

# Input data.
# We need to pass a train example and label to each cell or lstm model
train_data = list()
for _ in range(num_unrollings + 1):
    train_data.append(tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
train_inputs = train_data[:num_unrollings]
train_labels = train_data[1:]  # labels are inputs shifted by one time step.

# Unrolled LSTM loop.
# Pass to each cell a different training example
# Note that iteratively we pass the the last output (h_(t-1)) and state (c_(t-1)) to the new cell
outputs = list()
output = saved_output
state = saved_state
for i in train_inputs:
    output, state = lstm_cell(i, output, state)
    outputs.append(output)

# State saving across unrollings.
with tf.control_dependencies([saved_output.assign(output),
                              saved_state.assign(state)]):
    # Classifier.
    # in order to apply a fast classification and error propagation we concat each output
    # and put it in a simple vector. Each cell has an output of 64x64 (batch_size, hidden_size)
    # so tf.concat produce a vector if 640X64 (64(bath_size)x10 cells and 64 hidden_nodes)
    logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.concat(train_labels, 0), logits=logits))

print(tf.concat(train_labels, 0).shape)
print(logits.shape)
print(loss.shape)
# Optimizer.
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
    10.0, global_step, 5000, 0.1, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(0.5)
gradient_v = optimizer.compute_gradients(loss, tf.trainable_variables())
# gradients, _ = tf.clip_by_global_norm(gradient, 1.25)
# optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

# Predictions.
train_prediction = tf.nn.softmax(logits)


num_steps = 1001
summary_frequency = 100

with tf.Session() as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        batches = train_batches.next()
        feed_dict = dict()
        for i in range(num_unrollings + 1):
            feed_dict[train_data[i]] = batches[i]
        gr, l, predictions, lr = sess.run([loss, train_prediction, learning_rate, gradient_v], feed_dict=feed_dict)
        mean_loss += l
        if step % 10 == 0:
            for g, v in lr:
                if g is not None:
                    print("****************this is gradient*************")
                    print("gradient's shape:", g.shape)
                    print(g)
