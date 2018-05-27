import tensorflow as tf
from ops import conv2d, lrelu, bn, linear, se_layer
class Res_U_net(object):    
    
    def __init__(self, img, is_t, batch_size, uc, senet):
        
        self.img = img
        self.is_t = is_t
        self.batch_size=batch_size
        self.uc = uc
        self.senet = senet
    def discriminator(self):

        with tf.variable_scope('dis'):
            
            net=self.img
            
            for i in range(4):
                net = disc_block(net, 2**(i+6), self.is_t, self.batch_size, True, self.uc, self.senet)
                    
            net = linear(tf.reshape(net, [self.batch_size, -1]), 1, spectral_normed=True, update_collection=self.uc, name='linear')
            
            return net                
    
    def generator(self):
    
        with tf.variable_scope('gen'):
            
            skip = []
            net = conv2d(self.img, 16, k_h=1, k_w=1, name='conv_input')
            net = tf.nn.relu(bn(net, self.is_t, name='conv_input_bn_64'))
            skip.append(net)
            
            for i in range(5):
                net = encoder_block(net, 2**(i+4), self.is_t, self.batch_size, self.senet)
                skip.append(net)
            
            skip.reverse()
            
            for i in range(5):
                net = decoder_block(net, 2**(9-i), self.is_t, self.batch_size, skip[i+1], self.senet)
                    
            net = conv2d(net, 3, name='conv_output')
            
            return tf.nn.tanh(net) 
        
def encoder_block(img, dim, is_t, batch_size, senet):
        
    with tf.variable_scope('downsample'):
        
        im = conv2d(img, dim, k_h=1, k_w=1, name='conv1_1_{}'.format(dim))
        im = tf.nn.relu(bn(im, is_t=is_t, name='conv1_bn_1_{}'.format(dim)))
        im = conv2d(im, dim, name='conv1_2_{}'.format(dim))
        im = tf.nn.relu(bn(im, is_t=is_t, name='conv1_bn_2_{}'.format(dim)))
        im = conv2d(img, dim, k_h=1, k_w=1, name='conv1_3_{}'.format(dim))
        im = tf.nn.relu(bn(im, is_t=is_t, name='conv1_bn_3_{}'.format(dim)))
        im = img + im
        
        im_res = conv2d(im, dim*2, k_h=1, k_w=1, stride=2, name='conv_res_{}'.format(dim*2))
        im_res = tf.nn.relu(bn(im_res, is_t=is_t, name='conv_res_bn_{}'.format(dim*2)))
        
        im = conv2d(im, dim, k_h=1, k_w=1, name='conv2_1_{}'.format(dim))
        im = tf.nn.relu(bn(im, is_t=is_t, name='conv2_bn_1_{}'.format(dim)))
        im = conv2d(im, dim, name='conv2_2_{}'.format(dim))
        im = tf.nn.relu(bn(im, is_t=is_t, name='conv2_bn_2_{}'.format(dim)))
        if senet == True:
            im = se_layer(im, dim, 8, name='conv_se_{}'.format(dim))
        im = conv2d(img, dim*2, k_h=1, k_w=1, stride=2, name='con2_3_{}'.format(dim*2))
        im = tf.nn.relu(bn(im, is_t=is_t, name='conv2_bn_3_{}'.format(dim*2)))
        im = im_res + im
                
        return im    
        
def decoder_block(img, dim, is_t, batch_size, skip, senet):
        
    with tf.variable_scope('upsample'):
            
        h = skip.shape[1]
        w = skip.shape[2]

        im = tf.image.resize_bilinear(img, [h,w], name='bilinear')
        im = conv2d(im, dim, k_w=1, k_h=1, name='conv_bilinear_{}'.format(dim))  
        im = tf.nn.relu(bn(im, is_t=is_t, name='conv_bilinear_bn_{}'.format(dim)))
        im = conv2d(im, dim, name='conv_{}'.format(dim))
        im = tf.nn.relu(bn(im, name='conv_bn_{}'.format(dim)))
        if senet == True:
            im = se_layer(im, dim, 8, name='conv_se_{}'.format(dim))        
        im = tf.concat([im, skip], 3)       

        return im
    
def disc_block(img, dim, is_t, batch_size, sn, uc, senet):
    
        im_res = conv2d(img, dim*2, k_h=1, k_w=1, stride=2, spectral_normed=sn, update_collection=uc, name='conv_res_{}'.format(dim*2))
        im_res = lrelu(im_res)
    
        im = conv2d(img, dim, k_h=1, k_w=1, spectral_normed=sn, update_collection=uc, name='conv_1_{}'.format(dim))
        im = lrelu(im)
        im = conv2d(im, dim, spectral_normed=sn, update_collection=uc, name='conv_2_{}'.format(dim))
        im = lrelu(im)
        im = conv2d(im, dim*2, stride=2, k_h=1, k_w=1, spectral_normed=sn, update_collection=uc, name='conv_3_{}'.format(dim*2))
        im = lrelu(im)  
        if senet == True:
            im = se_layer(im, dim*2, 8, name='conv_se_{}'.format(dim*2))        
        im = im + im_res

        return im

    