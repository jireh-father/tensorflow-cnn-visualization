import tensorflow as tf


def summary_first_filters(weights, filter_summary_limit=3):
    x_min = tf.reduce_min(weights)
    x_max = tf.reduce_max(weights)
    weights_0_to_1 = (weights - x_min) / (x_max - x_min)
    # weights_0_to_255_uint8 = tf.image.convert_image_dtype(weights_0_to_1, dtype=tf.uint8)

    weights_transposed = tf.transpose(weights_0_to_1, [3, 0, 1, 2])
    tf.summary.image("filter_visualization", weights_transposed, int(filter_summary_limit))


def summary_feature_maps(inputs, input_ph, feature_maps, sess, batch_limit=3, feature_map_limit=3):
    units = sess.run(feature_maps, feed_dict={input_ph: inputs})
    for key in units:
        if len(units[key].shape) != 4:
            continue
        for i in range(units[key].shape[0]):
            for j in range(units[key].shape[3]):
                tf.summary.image("%s/%s/%d" % ('activation_visualization', key, j), units[key][i:i + 1, :, :, j:j + 1])
                if j + 1 >= feature_map_limit:
                    break
            if i + 1 >= batch_limit:
                break
