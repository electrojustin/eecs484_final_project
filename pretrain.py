import tensorflow as tf
import conv_helper
from tensorflow.examples.tutorials.mnist import input_data

def get_pretrain_params():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    #Build network
    W_conv1 = conv_helper.weight_variable([5, 5, 1, 32])
    b_conv1 = conv_helper.bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.leaky_relu(conv_helper.conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = conv_helper.max_pool_2x2(h_conv1)
    W_conv2 = conv_helper.weight_variable([5, 5, 32, 64])
    b_conv2 = conv_helper.bias_variable([64])
    h_conv2 = tf.nn.leaky_relu(conv_helper.conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = conv_helper.max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    W_fc1 = conv_helper.weight_variable([7 * 7 * 64, 1024])
    b_fc1 = conv_helper.bias_variable([1024])
    h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    W_fc2 = conv_helper.weight_variable([1024, 10])
    b_fc2 = conv_helper.bias_variable([10])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #Train for a number of iterations
        for i in range(5000):
            batch = mnist.train.next_batch(128)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    
        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

        #Dump the weights and biases from the partially trained network
        kernel_weights1 = tf.constant(W_conv1.eval())
        kernel_bias1 = tf.constant(b_conv1.eval())
        kernel_weights2 = tf.constant(W_conv2.eval())
        kernel_bias2 = tf.constant(b_conv2.eval())
        abstract_weights1 = tf.constant(W_fc1.eval())
        abstract_bias1 = tf.constant(b_fc1.eval())
        abstract_weights2 = tf.constant(W_fc2.eval())
        abstract_bias2 = tf.constant(b_fc2.eval())

    return kernel_weights1, kernel_bias1, kernel_weights2, kernel_bias2, abstract_weights1, abstract_bias1, abstract_weights2, abstract_bias2
