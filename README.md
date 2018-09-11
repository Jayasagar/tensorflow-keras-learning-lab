# tensorflow-keras-learning-lab
Learning examples on Tensorflow and Keras

# Setup Tensorflow
* change `Pipfile` python_version to `3.6`
* run command `pipenv --python 3.6`
#### Temporary solution to install Tensorflow using pipenv
`pipenv install --verbose https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.9.0-py3-none-any.whl``

### What are ConfigProto options
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                                        log_device_placement=True))
                                        
Whether soft placement is allowed. If allow_soft_placement is true, an op will be placed on CPU if
 1. there's no GPU implementation for the OP
 or
 2. no GPU devices are known or registered
or
  3. need to co-locate with reftype input(s) which are from CPU.

When log_device_placement=True, you will get a verbose output

