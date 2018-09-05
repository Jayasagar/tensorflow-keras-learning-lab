import tensorflow as tf

# Creates a node in graph
a = tf.constant(6, name = 'a')
b = tf.constant(12, name= 'b')

print('Node a:', a)

c = tf.add(a, b, name= 'c')

print('Node c:', c)

#  Two inout to TF graph flow
# TensorFlow uses tf.placeholder to handle inputs to the model.
# tf.placeholder lets you specify that some input will be coming in, of some shape and some type
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

z = tf.add(x, y, name= 'z')
p = tf.subtract(x, 1, name= 'p')
q = tf.multiply(z, p, name='q')


# define a session and run our computation graph:

with tf.Session() as session:
    feed_dict = {x: 6, y: 6}
    # pass data in and run the computation graph in a session
    output = session.run([q], feed_dict = feed_dict)
    print('TF Session computation output:', output)