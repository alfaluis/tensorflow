import tensorflow as tf
import numpy as np
import collections
import zipfile
import random
from sklearn.manifold import TSNE
from matplotlib import pylab

rand_uniform = tf.Variable(tf.random_uniform([10, 3], -1.0, 1.0))
embeddings = tf.Variable(tf.random_uniform([5, 3], -1.0, 1.0))
ids = tf.constant([4, 1])
embed = tf.nn.embedding_lookup(embeddings, ids)
with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    print(session.run([rand_uniform]))
    print(session.run([embeddings]))
    print(session.run([ids]))
    print(session.run([embed]))

# *************************************************************************************
# Check how skip-gram model actually work
# *************************************************************************************


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words):
    # return a list where the first element is the most common and the less common is the last element.
    # also the list is limited by the number inside the funcion most_common()
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    # dictionary where the most common word has value (index) 0 and the less common has value (index) 50000-1
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # assign to 'data' a index number to each word in the words variable (17MM), if it does not exist assign 0 index
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    # set unknown frequency as many as no indexing words were detected
    count[0][1] = unk_count
    # reverse dictionary where index is a key and a value is the word
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def generate_batch(data, batch_size, num_skips, skip_window):
    """ this function generate the batch (training data) and labels for the batch
    Ex:
        batch_size, num_skips, skip_window = 4, 2, 1
        # with these configurations the training data will have the form [n1,n2,n3,n4,n5,n6,n7,n8] and the way we create
        # the relationship between words [word1 (skip_num1) word2(skip_win) word3(skip_num2)]
        data = [10, 15, 13, 2, 20]  # ['the', 'cat', 'house', 'is', 'big']
        buffer = [10, 15, 13]  # ['the', 'cat', 'house']
        batch = [15, 15] -> labels = [10, 13]  # first iteration
        ...
        buffer = [15, 13, 2]
        batch = [13, 13] -> labels = [15, 2] # second iteration and final
        batch = [15, 15, 13, 13] -> labels = [10, 13, 15, 2]  # output batch

    :param batch_size: the lenght of the batch for training (as SGD)
    :param num_skips: Define the number of words that will have relationship with the center word in the buffer
    :param skip_window: Define the index for the center word in the buffer vector. Buffer 5x1 -> [x, x, skip, x, x]
    :return: batch (training data) and labels for the examples
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # create a batch of Vector(8,) and a labels of Matrix(8,1)
    # batch = np.ndarray(shape=batch_size, dtype=np.int32)
    batch = np.zeros(shape=batch_size, dtype=np.int32)
    # labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    labels = np.zeros(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    # create a special list with max buffer size of "span value"
    buffer = collections.deque(maxlen=span)
    # fill the buffer with the n°(span) elements from data (index of words)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        # print(buffer, data_index)
    # iterate as many times different words fit in batch (defined by batch_size)
    for i in range(batch_size // num_skips):
        # define which word is the center
        target = skip_window
        # define which word (index) must be avoid
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
                # print(target)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

# main variables for model
vocabulary_size = 20000
batch_size = 64
num_embedding = 128
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))

train_dataset = tf.placeholder(tf.int32, shape=None)
train_labels = tf.placeholder(tf.int32, shape=[None, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
# weight that going to work as features or word2vec (features generated automatically)
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, num_embedding], -1.0, 1.0))
# weight and bias for the softmax classifier
softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, num_embedding], stddev=1.0))
softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Return the features generated for the given training example
# embedding = [[.3, .5, .1],  and train = [[0],  --> embed = [.4, .1, .9]
#              [.4, .1, .9],               [1],
#              [.1, .9, .5]]               [0]]
embed = tf.nn.embedding_lookup(embeddings, train_dataset)
logits = tf.matmul(embed, tf.transpose(softmax_weights)) + softmax_biases
# convert train_labels from [batch_size, 1] to one hot -> [batch_size, vocabulary_size]
# train_labels = [[2],  and voc_size=4  --> labels_one_hot = [[0, 0, 1, 0],
#                 [1]]                                        [0, 1, 0, 0]]
labels_trans = tf.reshape(train_labels, shape=[-1])
labels_one_hot = tf.one_hot(indices=labels_trans, depth=vocabulary_size)
# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits))
optimizer = tf.train.AdagradOptimizer(0.2).minimize(loss)
# prediction model
prediction = tf.nn.softmax(tf.matmul(embed, tf.transpose(softmax_weights)) + softmax_biases)
# Create a normalize vector to measure the distant between it
# equation: cos_dist = V_embed / norm(V_embed) -> Remember that embedded is [vocabulary, features]
# so norm size must be [vocabulary, 1]
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

# create session and train the word2vec skip-gram
filename = 'text8.zip'
words = read_data(filename)
print('Data size %d' % len(words))

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.


total_loss = 0.0
data_index = 0
num_step = 20001
with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    for step in range(num_step):

        batch_data, batch_labels = generate_batch(data, batch_size, 2, 1)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        l, op, feature, norm2 = session.run([loss, optimizer, embeddings, norm], feed_dict=feed_dict)
        total_loss += l
        if step % 500 == 0:
            print('Step: {0:5d}, loss actual: {1:.2f}'.format(step, total_loss / 500))
            total_loss = 0
        if step % 1000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    pred = prediction.eval(feed_dict={train_dataset: [400, 1000]})
    print(np.argmax(pred[0]), np.argmax(pred[1]))

num_points = 400
final_embeddings = feature/norm2
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15,15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
    pylab.show()

words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)

# test for one hot transformation
idx = tf.placeholder(tf.int32, shape=[4])
one = tf.one_hot(indices=idx, depth=5)
