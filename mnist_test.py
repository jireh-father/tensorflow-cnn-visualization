from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import mnist_loader as loader
import visualizer
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_path', "/tmp/summary", "summary path")

image_size = 28
image_channel = 1
label_cnt = 10
num_epochs = 10
batch_size = 64

inputs = tf.placeholder(tf.float16, [batch_size, image_size, image_size, image_channel])
labels = tf.placeholder(tf.int64, [batch_size, ])

conv1_weights = tf.Variable(tf.truncated_normal([5, 5, image_channel, 32], stddev=0.1, dtype=tf.float16))
visualizer.summary_first_filters(conv1_weights)
conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float16))
conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, dtype=tf.float16))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float16))
fc1_weights = tf.Variable(tf.truncated_normal([7 * 7 * 64, 512], stddev=0.1, dtype=tf.float16))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float16))
fc2_weights = tf.Variable(tf.truncated_normal([512, label_cnt], stddev=0.1, dtype=tf.float16))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[label_cnt], dtype=tf.float16))

end_points = {}
conv = tf.nn.conv2d(inputs, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
end_points["conv1"] = conv

pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
end_points["pool1"] = pool

conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
end_points["conv2"] = conv

pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
end_points["pool2"] = pool

pool_shape = pool.get_shape().as_list()
reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
hidden = tf.nn.dropout(hidden, 0.5)
logits = tf.matmul(hidden, fc2_weights) + fc2_biases
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
loss += 5e-4 * regularizers

optimizer = tf.train.MomentumOptimizer(0.1, 0.9).minimize(loss)

train_prediction = tf.nn.softmax(logits)

train_data, train_labels, validation_data, validation_labels = loader.load_mnist()
train_size = train_labels.shape[0]
start_time = time.time()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in xrange(int(num_epochs * train_size) // batch_size):
        offset = (step * batch_size) % (train_size - batch_size)
        batch_data = train_data[offset:(offset + batch_size), ...]
        batch_labels = train_labels[offset:(offset + batch_size)]
        feed_dict = {inputs: batch_data, labels: batch_labels}
        loss_result, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        if step % 100 == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Step %d (epoch %.2f), %.1f ms' %
                  (step, float(step) * batch_size / train_size,
                   1000 * elapsed_time / 100))
            print('Minibatch loss: %.3f' % loss_result)

    visualizer.summary_feature_maps(validation_data[0:batch_size], inputs, end_points, sess)

    merged = tf.summary.merge_all()
    if not os.path.isdir(FLAGS.summary_path):
        os.makedirs(FLAGS.summary_path)
    writer = tf.summary.FileWriter(FLAGS.summary_path)
    summary = sess.run(merged, feed_dict={inputs: validation_data[0:16]})
    writer.add_summary(summary, 0)
    writer.close()
