import tensorflow as tf

n_input_nodes = 2
n_output_nodes = 1
#Inputs
x = tf.placeholder(tf.float32, (None, n_input_nodes))
# Weights
# tf.ones(
#     shape,(2, 1) => 2 dimensions and length of each dimension
#     dtype=tf.float32,
#     name=None
# )
W = tf.Variable(tf.ones((n_input_nodes, n_output_nodes)), dtype=tf.float32)
# Bias = Threshold
b = tf.Variable(tf.zeros(n_output_nodes), dtype=tf.float32)

z = tf.matmul(x, W)
out = tf.sigmoid(z)

# close the Session automatically again after printing the output
with tf.Session() as session:
    # Only after running tf.global_variables_initializer() in a session
    # then only  your variables hold the values we told them to hold
    # when you declare them (tf.Variable(tf.zeros(...)), tf.Variable(tf.random_normal(...)),...).
    tf.global_variables_initializer().run(session=session)
    feed_dict = {x: [[0.25, 0.15]]}
    output = session.run(out, feed_dict=feed_dict)
    print('Values in Weight:', session.run(W))
    print('output:', output)