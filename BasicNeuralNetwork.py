import tensorflow as tf

n_input_nodes = 2
n_output_nodes = 1
#Inputs
x = tf.placeholder(tf.float32, (None, n_input_nodes))
# Weights
W = tf.Variable(tf.ones((n_input_nodes, n_output_nodes)), dtype=tf.float32)
# Bias = Threshold
b = tf.Variable(tf.zeros(n_output_nodes), dtype=tf.float32)

z = tf.matmul(x, W)
out = tf.sigmoid(z)

with tf.Session() as session:
    tf.global_variables_initializer().run(session=session)
    feed_dict = {x: [[0.25, 0.15]]}
    output = session.run(out, feed_dict=feed_dict)

    print('output:', output)