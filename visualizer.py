import tensorflow as tf


def convert_weights(weights):
    x_min = tf.reduce_min(weights)
    x_max = tf.reduce_max(weights)
    weights_0_to_1 = (weights - x_min) / (x_max - x_min)
    # weights_0_to_255_uint8 = tf.image.convert_image_dtype(weights_0_to_1, dtype=tf.uint8)

    return tf.transpose(weights_0_to_1, [3, 0, 1, 2])


def summary_filter(weights, filter_summary_limit=3):
    weights_transposed = convert_weights(weights)
    tf.summary.image("filter_visualization", weights_transposed, int(filter_summary_limit))


def summary_filters(weights_list, batch_limit=3, feature_map_limit=3):
    for weight_idx, weights in enumerate(weights_list):
        weights_transposed = convert_weights(weights)
        summary_name = "filter_visualization/layer%d" % weight_idx
        for i in range(weights_transposed.shape[0]):
            if weights_transposed.shape[3] == 3:
                tf.summary.image("%s/%d" % (summary_name, i), weights_transposed[i:i + 1, :, :, :])
            else:
                for j in range(weights_transposed.shape[3]):
                    tf.summary.image("%s/%d/filter%d" % (summary_name, i, j),
                                     weights_transposed[i:i + 1, :, :, j:j + 1])
                    if j + 1 >= feature_map_limit:
                        break
            if i + 1 >= batch_limit:
                break


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
