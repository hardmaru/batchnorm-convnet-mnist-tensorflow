# self contained dummy skeleton script for testing and training mnist convnet+batchnorm based classifier
# useful for other projects going forward as template

import numpy as np
import tensorflow as tf
from mnist_data import *

class batch_norm(object):
  """Code modification of http://stackoverflow.com/a/33950177"""
  def __init__(self, batch_size, epsilon=1e-5, momentum = 0.1, name="batch_norm"):
    with tf.variable_scope(name) as scope:
      self.epsilon = epsilon
      self.momentum = momentum
      self.batch_size = batch_size

      self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
      self.name=name

  def __call__(self, x, train=True):
    shape = x.get_shape().as_list()

    with tf.variable_scope(self.name) as scope:
      self.gamma = tf.get_variable("gamma", [shape[-1]],
                        initializer=tf.random_normal_initializer(1., 0.02))
      self.beta = tf.get_variable("beta", [shape[-1]],
                        initializer=tf.constant_initializer(0.))
      self.mean, self.variance = tf.nn.moments(x, [0, 1, 2])

      return tf.nn.batch_norm_with_global_normalization(
        x, self.mean, self.variance, self.beta, self.gamma, self.epsilon,
        scale_after_normalization=True)

def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                             tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv

class Model():
  def __init__(self, batch_size=100, x_dim = 26, y_dim = 26, learning_rate= 0.001, df_dim = 32):
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.num_class = 10
    self.df_dim = df_dim

    # tf Graph batch of image (batch_size, height, width, depth)
    self.batch = tf.placeholder(tf.float32, [batch_size, x_dim, y_dim, 1])
    self.batch_label = tf.placeholder(tf.float32, [batch_size, self.num_class]) # mnist labels for the batch

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(batch_size, name='d_bn1')
    self.d_bn2 = batch_norm(batch_size, name='d_bn2')

    self.predict = self.discriminator(self.batch)
    self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.batch_label * tf.log(self.predict), reduction_indices=[1]))
    correct_prediction = tf.equal(tf.argmax(self.predict,1), tf.argmax(self.batch_label,1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

    # Initializing the tensor flow variables
    init = tf.initialize_all_variables()

    # Launch the session
    self.sess = tf.InteractiveSession()
    self.sess.run(init)
    self.saver = tf.train.Saver(tf.all_variables())

  def discriminator(self, image, reuse=False):
    # converted to a classifier, similar to dcgan discriminiator but with softmax
    if reuse:
        tf.get_variable_scope().reuse_variables()

    h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
    h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
    h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
    h3 = linear(tf.reshape(h2, [self.batch_size, -1]), self.num_class, 'd_h2_lin')

    return tf.nn.softmax(h3)

  def to_one_hot(self, label):
    # convert labels, a numpy list of labels (of size batch_size) to the one hot equivalent
    return np.eye(self.num_class)[label]

  def partial_train(self, batch, label):
    _, loss, accuracy = self.sess.run((self.train_op, self.cross_entropy, self.accuracy),
                              feed_dict={self.batch: batch, self.batch_label: self.to_one_hot(label)})
    return loss, accuracy

  def save_model(self, checkpoint_path, epoch):
    """ saves the model to a file """
    self.saver.save(self.sess, checkpoint_path, global_step = epoch)

  def load_model(self, checkpoint_path):
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print "loading model: ",ckpt.model_checkpoint_path
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)

def main():
  model = Model()
  mnist = read_data_sets()
  batch_size = model.batch_size
  num_examples = mnist.num_examples
  training_epochs = 5
  checkpoint_path = os.path.join('save', 'model.ckpt')

  # load previously trained model if appilcable
  ckpt = tf.train.get_checkpoint_state('save')
  if ckpt:
    model.load_model(dirname)

  # Training cycle
  for epoch in range(training_epochs):
    avg_loss = 0.
    avg_accuracy = 0.
    mnist.shuffle_data()
    total_batch = int(num_examples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
      batch_images, batch_labels = mnist.next_batch(batch_size, with_label = True) # obtain training labels

      loss, accuracy = model.partial_train(batch_images, batch_labels)

      assert( loss < 1000000 ) # make sure it is not NaN or Inf
      assert( accuracy < 1000000 ) # make sure it is not NaN or Inf

      # Display logs per epoch step
      if (i % 100 == 0):
        print "epoch:", '%04d' % (epoch), \
              "batch:", '%04d' % (i), \
              "loss=", "{:.6f}".format(loss), \
              "accuracy=", "{:.6f}".format(accuracy)

      # Compute average loss
      avg_loss += loss / num_examples * batch_size
      avg_accuracy += accuracy / num_examples * batch_size

    # Display logs per epoch step
    print "epoch:", '%04d' % (epoch), \
          "avg_loss=", "{:.6f}".format(avg_loss), \
          "avg_accuracy=", "{:.6f}".format(avg_accuracy)

    # save model every 1 epochs
    if epoch >= 0 and epoch % 1 == 0:
      model.save_model(checkpoint_path, epoch)
      print "model saved to {}".format(checkpoint_path)

  # save model one last time, under zero label to denote finish.
  model.save_model(checkpoint_path, 0)

if __name__ == '__main__':
  main()
