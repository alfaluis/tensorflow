import tensorflow as tf

# create two nodes or Tensors the constant type
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
# it just print the type of data create and not the values
print(node1, node2)

# to evaluate this Tensors we must run the computational graph
session = tf.Session()
print(session.run([node1, node2]))

# creating more complicated graph
# combining two Tensors
node3 = tf.add(node1, tf.constant([3.0, 5.0]))
node3 = tf.add(node1, node2)
print(session.run(node3))

"""
node1 --
        |
        |-- Add() -->
        |
node2 --
"""

# graph accepting external parameters
n1 = tf.placeholder(tf.float32, name='n1')
n2 = tf.placeholder(tf.float32, name='n2')
n3 = tf.add(n1, n2, name='sum')
tf.summary.scalar('result', n3)
summ = tf.summary.merge_all()
parameters = {n1: 5.0, n2: 10.0}
parameters2 = {n1: [5, 6], n2: [10, 6]}
writer = tf.summary.FileWriter('test_file1')
writer.add_graph(session.graph)

[res, s] = session.run([n3, summ], feed_dict={n1: 5, n2: 10})
writer.add_summary(s, 1)

print()
print(session.run(n3, parameters2))
# or we can use more simple syntax
adder_and_mult = (n1 + n2) * 10
# or
adder_and_mult = n3 * 10
print(session.run(adder_and_mult, parameters))

"""
n1 (param) --       10 --    
             |           |
             |--- n3() --|-- adder_and_mult() --> Result 
             |
n2 (param) --
"""

# Adding trainable parameters to the graph model
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# now it is necessary to initialize the tf.Variable (explicitly)
init = tf.global_variables_initializer()
session.run(init)

# evaluate our trainable model
print(session.run(linear_model, {x: [1, 2, 3, 4]}))

# adding a lost function
y = tf.placeholder(tf.float32)
squared_delta = tf.square(y - linear_model)
loss = tf.reduce_sum(squared_delta)
print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
"""
it is the same:
y_pred = [ 0.        ,  0.30000001,  0.60000002,  0.90000004]
loss = sum((y - y_pred) ** 2)
"""

session = tf.Session()
# train a model
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
squared_delta = tf.square(y - linear_model)
loss = tf.reduce_sum(squared_delta)
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss=loss)
init = tf.global_variables_initializer()
session.run(init)
# adjust values
for i in range(1000):
    v, w1, b1,l = session.run([optimizer, W, b, loss], {x: x_train, y: y_train})
    if i % 100 == 0:
        print(l, w1, b1)

print(session.run([W, b, loss], {x: x_train, y: y_train}))

# make code more readable. One way to create more readable code is create graph structures as follow
# in this example we create a graph object and later we define it as default in order to pass operations
g = tf.Graph()
with g.as_default():
  c = tf.constant(5.0)
  assert c.graph is g

sess = tf.Session(graph=g)
print(sess.run(c))

# here is the same last code but using graph
g1 = tf.Graph()
with g1.as_default():
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    squared_delta = tf.square(y - linear_model)
    loss = tf.reduce_sum(squared_delta)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss=loss)
    init = tf.global_variables_initializer()

# and here we call the code as always
session = tf.Session(graph=g1)
session.run(init)
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
for i in range(1000):
    v, w1, b1,l = session.run([optimizer, W, b, loss], {x: x_train, y: y_train})
    if i % 100 == 0:
        print(l, w1, b1)
v, w1, b1, l = session.run([optimizer, W, b, loss], {x: x_train, y: y_train})
# it is important close the session
session.close()

# there is another option to create a session using with command which ensure that the session will be closed
# once it was used

with tf.Session(graph=g1) as session:
    session.run(init)
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]
    for i in range(1000):
        v, w1, b1, l = session.run([optimizer, W, b, loss], {x: x_train, y: y_train})
        if i % 100 == 0:
            print(l, w1, b1)
    v, w1, b1, l = session.run([optimizer, W, b, loss], {x: x_train, y: y_train})
