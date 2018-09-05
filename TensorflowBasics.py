import tensorflow as tf

# Creates a node in graph
a = tf.constant(6, name = 'a')
b = tf.constant(12, name= 'b')

print('Node a:', a)

c = tf.add(a, b, name= 'c')

print('Node c:', c)