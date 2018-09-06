'''
 When you enable Eager execution, TensorFlow operations execute immediately as they're called from Python.
 That means you do not execute a pre-constructed graph with Session.run().
 This allows for fast debugging and a more intuitive way to get started with TensorFlow.
'''
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

import tensorflow as tf

