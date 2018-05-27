from tensorflow.contrib.layers.python.layers import batch_norm
import tensorflow as tf
import numpy as np

def bn(x , is_t=True, name="scope" , reuse=False):
    return batch_norm(x , epsilon=1e-5, decay=0.9 , is_training=is_t, scale=True, scope=name , reuse= reuse , updates_collections=None)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)
   
def conv2d(input_, output_dim,  k_h=3, k_w=3, stride=1, stddev=0.02, spectral_normed=False, update_collection=None, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if spectral_normed:
            conv = tf.nn.conv2d(input_, spectral_normed_weight(w, update_collection=update_collection),
                          strides=[1, stride, stride, 1], padding='SAME')            
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        
        return conv

def deconv2d(input_, output_shape,k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    
    output_shape = np.array(output_shape).astype(np.int32)
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
        
def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
  # Usually num_iters = 1 will be enough
  W_shape = W.shape.as_list()
  W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
  if u is None:
    u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
  def power_iteration(i, u_i, v_i):
    v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
    u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
    return i + 1, u_ip1, v_ip1
  _, u_final, v_final = tf.while_loop(
    cond=lambda i, _1, _2: i < num_iters,
    body=power_iteration,
    loop_vars=(tf.constant(0, dtype=tf.int32),
               u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
  )
  if update_collection is None:
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    with tf.control_dependencies([u.assign(u_final)]):
      W_bar = tf.reshape(W_bar, W_shape)
  else:
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    W_bar = tf.reshape(W_bar, W_shape)
    # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
    # has already been collected on the first call.
    if update_collection != 'NO_OPS':
      tf.add_to_collection(update_collection, u.assign(u_final))
  if with_sigma:
    return W_bar, sigma
  else:
    return W_bar
    
def linear(input_, output_size, name=None, spectral_normed=False, update_collection=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        
        if spectral_normed:
          matrix = tf.matmul(input_, spectral_normed_weight(matrix, update_collection=update_collection))
        else:
          matrix = tf.matmul(input_, matrix)  
          
        if with_w:
            return matrix + bias, matrix, bias
        else:
            return matrix + bias    
        
def se_layer(net, out_dim, ratio, name):
    with tf.name_scope(name) :
                
        feature_size = tf.shape(net)

        # global average pooling
        squeeze = tf.reduce_mean(net, [1, 2], name=name+'_squeeze_global_pool', keep_dims=True)
        squeeze = conv2d(squeeze, output_dim=256, k_w=1, k_h=1, name=name+"_squeeze_conv")
        squeeze = tf.image.resize_bilinear(squeeze, (feature_size[1], feature_size[2]))

        excitation = tf.layers.dense(squeeze, use_bias=True, units=out_dim / ratio, name=name+'_fc1')
        excitation = tf.nn.relu(excitation)
        excitation = tf.layers.dense(excitation, use_bias=True, units=out_dim, name=name+'_fc2')
        excitation = tf.nn.sigmoid(excitation)

        scale = net * excitation
        
        return scale
    
def concat(x, y, name):
    
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    z = []
    if x_shapes[1] > y_shapes[1]:
        size = int(x_shapes[1] - y_shapes[1])
        x = tf.unstack(x, axis=2)
        for i in range(len(x) - size):
            z.append(x[i])
        x = tf.stack(z, axis=2)
        z.clear()
    if x_shapes[2] > y_shapes[2]:
        size = int(x_shapes[2] - y_shapes[2])
        x = tf.unstack(x, axis=1)
        for i in range(len(x) - size):
            z.append(x[i])
        x = tf.stack(z, axis=1)
        z.clear()
        
    return tf.concat([x, y], 3)
