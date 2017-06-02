# tensorflow-cnn-visualization
Visualizing cnn feature maps and filters on tensorboard.

## mnist visualization

### conv layers
![alt text](https://raw.githubusercontent.com/jireh-father/tensorflow-cnn-visualization/master/img/feature_map_visualization_conv.jpg)

### pooling layers
![alt text](https://raw.githubusercontent.com/jireh-father/tensorflow-cnn-visualization/master/img/feature_map_visualization_pooling.jpg)

### filters
![alt text](https://raw.githubusercontent.com/jireh-father/tensorflow-cnn-visualization/master/img/filter_visualization.jpg)


## Usage

### Visualize feature maps(activations) on TensorBoard

summary_feature_maps(data, input_op, feature_maps, sess, batch_limit=3, feature_map_limit=3)

```
import visualizer

...

inputs = tf.placeholder(tf.float16, [batch_size, image_size, image_size, image_channel])

conv1_weights = tf.Variable(tf.truncated_normal([5, 5, image_channel, 32], stddev=0.1, dtype=tf.float16))
conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float16))
conv = tf.nn.conv2d(inputs, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
end_points["conv1"] = conv
pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
end_points["pool1"] = pool

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    ...

    visualizer.summary_feature_maps(validation_sample_data, inputs, end_points, sess)

    ...

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.summary_path)
    summary = sess.run(merged, feed_dict={inputs: validation_sample_data})
    writer.add_summary(summary, 0)
    writer.close()
```

### Visualize filters(weights, kernels) on TensorBoard

summary_filter(filters, filter_summary_limit=3):
summary_filters(filter_list, layer_input_limit=3, layer_output_limit=3)

```
import visualizer

...

conv1_weights = tf.Variable(tf.truncated_normal([5, 5, image_channel, 32], stddev=0.1, dtype=tf.float16))
conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float16))
conv = tf.nn.conv2d(inputs, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

visualizer.summary_filters([conv1_weights, conv2_weights])

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    ...

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.summary_path)
    summary = sess.run(merged, feed_dict={inputs: validation_sample_data})
    writer.add_summary(summary, 0)
    writer.close()
```


## Test

```
python mnist_test.py --summary_path=yourpath

# after training
cd yourpath

tensorboard --logdir=./ --host=0.0.0.0
```