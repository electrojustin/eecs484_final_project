#Main file

import tensorflow as tf
import pretrain
import random
import conv_helper
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
from sklearn.cluster import KMeans

num_categories = 50 #CHANGE ME

y_ = tf.placeholder(tf.float32, [None, 10])
x = tf.placeholder(tf.float32, [None, 784])

def create_base_model(kernel_weights1, kernel_bias1, kernel_weights2, kernel_bias2, abstract_weights1, abstract_bias1, abstract_weights2, abstract_bias2):
    #Builds the network again
    
    #We're re-training the convolution layers, so make these variables
    W_conv1 = tf.Variable(kernel_weights1)
    b_conv1 = tf.Variable(kernel_bias1)
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.leaky_relu(conv_helper.conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = conv_helper.max_pool_2x2(h_conv1)
    W_conv2 = tf.Variable(kernel_weights2)
    b_conv2 = tf.Variable(kernel_bias2)
    h_conv2 = tf.nn.leaky_relu(conv_helper.conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = conv_helper.max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    #We're fixing the abstract layers, so keep these constants
    W_fc1 = abstract_weights1
    b_fc1 = abstract_bias1
    h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    W_fc2 = abstract_weights2
    b_fc2 = abstract_bias2
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_conv


def cluster_data(dataset, labels):
    #Clusters the dataset and the labels

    #Train clustering algorithm
    kmeans = KMeans(n_clusters=num_categories, n_jobs=-1).fit(dataset)

    #Separate the training dataset
    clustered_data = []
    clustered_labels = []
    for i in range(0, num_categories):
        cluster = []
        current_labels = []
        for j in range(0, len(dataset)):
            if kmeans.labels_[j] == i:
                cluster.append(dataset[j])
                current_labels.append(labels[j])

        clustered_data.append(cluster)
        clustered_labels.append(current_labels)

    return clustered_data, clustered_labels

def get_batch(index_pop, data, labels, batch_size):
    #Samples a dataset and its labels without replacement to create a batch

    batch_indices = random.sample(index_pop, batch_size)

    ret_data = []
    ret_labels = []
    for index in batch_indices:
        ret_data.append(data[index])
        ret_labels.append(labels[index])

    return ret_data, ret_labels

kernel_weights1, kernel_bias1, kernel_weights2, kernel_bias2, abstract_weights1, abstract_bias1, abstract_weights2, abstract_bias2 = pretrain.get_pretrain_params()
print("Pre-training complete!")

clustered_data, clustered_labels = cluster_data(mnist.train.images, mnist.train.labels)
clustered_test_data, clustered_test_labels = cluster_data(mnist.test.images, mnist.test.labels)
print("Clustering complete!")

#Fine tune the models
saver = tf.train.Saver()
for i in range(0, num_categories):
    index_pop = range(0, len(clustered_data[i]))

    y_conv = create_base_model(kernel_weights1, kernel_bias1, kernel_weights2, kernel_bias2, abstract_weights1, abstract_bias1, abstract_weights2, abstract_bias2)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for j in range(5000):
            batch_data, batch_labels = get_batch(index_pop, clustered_data[i], clustered_labels[i], 64)
            if j % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_data, y_: batch_labels})
                print('model %d, step %d, training accuracy %g' % (i, j, train_accuracy))

            train_step.run(feed_dict={x: batch_data, y_: batch_labels})


        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print('model %d, test accuracy %g' % (i, accuracy.eval(feed_dict={x: clustered_test_data[i], y_: clustered_test_labels[i]})))
        print('dumping model')

        saver.save(sess, 'fine_tuned_models/model' + str(i) + '.ckpt')
