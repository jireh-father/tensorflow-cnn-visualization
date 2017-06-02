import tensorflow as tf


def summary_filter(filters, filter_summary_limit=3):
    tf.summary.image("filter_visualization", tf.transpose(filters, [3, 0, 1, 2]), int(filter_summary_limit))


def summary_filters(filter_list, layer_input_limit=3, layer_output_limit=3):
    for filter_idx, filters in enumerate(filter_list):
        filters = tf.transpose(filters, [3, 0, 1, 2])
        summary_name = "filter_visualization_layer%d" % filter_idx
        for layer_output_idx in range(filters.shape[0]):
            if layer_output_idx >= layer_output_limit:
                break
            if filters.shape[3] == 3:
                tf.summary.image("%s/%d" % (summary_name, layer_output_idx),
                                 filters[layer_output_idx:layer_output_idx + 1, :, :, :])
            else:
                for layer_input_idx in range(filters.shape[3]):
                    if layer_input_idx >= layer_input_limit:
                        break
                    tf.summary.image("%s/%d/filter%d" % (summary_name, layer_output_idx, layer_input_idx),
                                     filters[layer_output_idx:layer_output_idx + 1, :, :,
                                     layer_input_idx:layer_input_idx + 1])


def summary_feature_maps(data, input_op, feature_maps, sess, batch_limit=3, feature_map_limit=3):
    units = sess.run(feature_maps, feed_dict={input_op: data})
    for key in units:
        if len(units[key].shape) != 4:  # only use feature maps
            continue
        for i in range(units[key].shape[0]):
            if i >= batch_limit:
                break
            for j in range(units[key].shape[3]):
                if j >= feature_map_limit:
                    break
                tf.summary.image("%s_%s/%d" % ('feature_map_visualization', key, j), units[key][i:i + 1, :, :, j:j + 1])
