from tensorflow.contrib.layers.python.layers import batch_norm
import tensorflow as tf
import numpy as np


def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 2, 4, 3])
    
    
    return tf.reshape(X, shape_2)

def pixelShuffler(inputs, scale=2):
    size = inputs.shape
    batch_size = int(size[0])
    h = int(size[1])
    w = int(size[2])
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale) #512/4 = 128
    channel_factor = c // channel_target  #512/128 = 4 

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale] # 10,8,8,2,2
    shape_2 = [batch_size, h * scale, w * scale, 1] #10,16,16,1
    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)#10,8,8,4
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output

def bn(x , name="scope" , is_t=True, reuse=False):
    return batch_norm(x , epsilon=1e-5, decay=0.9 , is_training=is_t, scale=True, scope=name , reuse= reuse , updates_collections=None)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)
    
def pixel_dcl(inputs, output_dim, k_h=3, k_w=3, name='pixel_dcl', d_format='NHWC'):

    axis = (d_format.index('H'), d_format.index('W'))
    conv0 = conv2d(inputs, output_dim,  k_h=2, k_w=2, name=name+'_conv2')
    conv1 = conv2d(conv0 , output_dim,  k_h=2, k_w=2, name=name+'_conv1')
    dilated_conv0 = dilate_tensor(conv0, axis, (0, 0), name+'/dialte_conv0')
    dilated_conv1 = dilate_tensor(conv1, axis, (1, 1), name+'/dialte_conv1')
    conv1 = tf.add(dilated_conv0, dilated_conv1, name+'/add1')
    with tf.variable_scope(name+'/conv2'):
        shape = list([k_h, k_w]) + [output_dim, output_dim]
        weights = tf.get_variable(
            'weights', shape, initializer=tf.truncated_normal_initializer())
        weights = tf.multiply(weights, get_mask(shape, name))
        strides = [1, 1, 1, 1]
        conv2 = tf.nn.conv2d(conv1, weights, strides, padding='SAME',
                             data_format=d_format)
    outputs = tf.add(conv1, conv2, name=name+'/add2')

    return outputs

def get_mask(shape, scope):
    new_shape = (np.prod(shape[:-2]), shape[-2], shape[-1])
    mask = np.ones(new_shape, dtype=np.float32)
    for i in range(0, new_shape[0], 2):
        mask[i, :, :] = 0
    mask = np.reshape(mask, shape, 'F')
    return tf.constant(mask, dtype=tf.float32, name=scope+'/mask')

def dilate_tensor(inputs, axes, shifts, scope):
    for index, axis in enumerate(axes):
        eles = tf.unstack(inputs, axis=axis, name=scope+'/unstack%s' % index)
        zeros = tf.zeros_like(
            eles[0], dtype=tf.float32, name=scope+'/zeros%s' % index)
        for ele_index in range(len(eles), 0, -1):
            eles.insert(ele_index-shifts[index], zeros)
        inputs = tf.stack(eles, axis=axis, name=scope+'/stack%s' % index)
    return inputs
   
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
        
def l2_norm(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_normed_weight(w, num_iters=1, update_collection=None):

  w_shape = w.shape.as_list()
  w = tf.reshape(w, [-1, w_shape[-1]])

  u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
  
  u_hat = u
  v_hat = None
  for i in range(num_iters):
      v_ = tf.matmul(u_hat, tf.transpose(w))
      v_hat = l2_norm(v_)

      u_ = tf.matmul(v_hat, w)
      u_hat = l2_norm(u_)
       
  if update_collection is None:
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))[0, 0]
    w_bar = w / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
      w_bar = tf.reshape(w_bar, w_shape)
  else:
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))[0, 0]
    w_bar = w / sigma
    w_bar = tf.reshape(w_bar, w_shape)
    if update_collection != 'NO_OPS':
      tf.add_to_collection(update_collection, u.assign(u_hat))

  return w_bar
    
def linear(input_, output_size, scope=None, spectral_normed=False, update_collection=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
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
def spectral_norm(w, iteration=1):
   w_shape = w.shape.as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])

   u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

   u_hat = u
   v_hat = None
   for i in range(iteration):
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = l2_norm(v_)

       u_ = tf.matmul(v_hat, w)
       u_hat = l2_norm(u_)

   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
   w_norm = w / sigma

   with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = tf.reshape(w_norm, w_shape)

   return w_norm
    
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
