import tensorflow as tf
from ops import linear, conv2d, lrelu, bn, concat, se_layer
slim = tf.contrib.slim
class DeeplabV3_Plus(object):

    def __init__(self, img, is_t, batch_size, uc, senet):
        
        self.img = img
        self.is_t = is_t
        self.batch_size=batch_size
        self.uc = uc        
        self.senet = senet
    
    def discriminator(self):

        with tf.variable_scope('dis'):
         
            net = disc_block(self.img, self.is_t, self.batch_size, True, self.uc, self.senet)
            net = linear(tf.reshape(net, [self.batch_size, -1]), 1, spectral_normed=True, update_collection=self.uc, name='linear')    
           
            return net 

    def generator(self):
    
        with tf.variable_scope("generator"):
                
            net, skip = encoder_block(self.img, self.is_t, self.senet)     
            net = atrous_spatial_pyramid_pooling(net, scope="gen_aspp", depth=256)  
            net = decoder_block(net, skip, self.is_t)

            return tf.nn.tanh(net) 

def encoder_block(img, is_t, senet):
        
    with tf.variable_scope('gen_downsample'):
        with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1):
                
            skip = []
            im = conv2d(img, output_dim=16, k_w=1, k_h=1, name='conv0_16')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv0_bn_16'))
            skip.append(im)
            # block1 output_size=128
            im = conv2d(im, output_dim=32, stride=2, name='conv1_32')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv1_bn_32'))
            im = conv2d(im, output_dim=64, name='conv1_64')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv1_bn_64'))
            im_res = conv2d(im, output_dim=128, k_h=1, k_w=1, stride=2, name='conv1_res_128')
            im_res = bn(im_res, is_t=is_t, name='conv1_res_bn_128')
            skip.append(im)
            # block2 output_size=64
            im = slim.separable_conv2d(im, 128, [3,3], scope='conv2_128')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv2_bn_128'))
            im = slim.separable_conv2d(im, 128, [3,3], scope='conv2_128_2')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv2_bn_128_2'))
            im = slim.separable_conv2d(im, 128, [3,3], stride=2, scope='conv2_128_3')
            im = bn(im, is_t=is_t, name='conv2_bn_128_3')
            if senet == True:
                im = se_layer(im, 128, 8, name='conv_se_128')
            im = tf.add(im, im_res)
            im_res = conv2d(im, output_dim=256, k_h=1, k_w=1, stride=2, name='conv2_res_256')
            im_res = bn(im_res, is_t=is_t, name='conv2_res_bn_256')
            skip.append(im)
            #block3 output_size=32
            im = tf.nn.relu(im)
            im = slim.separable_conv2d(im, 256, [3,3], scope='conv3_256')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv3_bn_256'))
            im = slim.separable_conv2d(im, 256, [3,3], scope='conv3_256_2')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv3_bn_256_2'))
            im = slim.separable_conv2d(im, 256, [3,3], stride=2, scope='conv3_256_3')
            im = bn(im, is_t=is_t, name='conv3_bn_256_3')
            if senet == True:
                im = se_layer(im, 256, 8, name='conv_se_256')
            im = tf.add(im, im_res)
            im_res = conv2d(im, output_dim=728, k_h=1, k_w=1, stride=2, name='conv3_res_728')
            im_res = bn(im_res, is_t=is_t, name='conv3_res_bn_728')  
            skip.append(im)
            #block4 output_size=16
            im = tf.nn.relu(im)
            im = slim.separable_conv2d(im, 728, [3,3], scope='conv4_728')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv4_bn_728'))
            im = slim.separable_conv2d(im, 728, [3,3], scope='conv4_728_2')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv4_bn_728_2'))
            im = slim.separable_conv2d(im, 728, [3,3], stride=2, scope='conv4_728_3')
            im = bn(im, is_t=is_t, name='conv4_bn_728_3')
            if senet == True:
                im = se_layer(im, 728, 8, name='conv_se_728')
            im = tf.add(im, im_res) 
            skip.append(im)
            
            #Middle flow
            for i in range(8):
                im_res = im
                im = tf.nn.relu(im)
                im = slim.separable_conv2d(im, 728, [3,3], scope='conv_mf_sp1_{}'.format(i))
                im = tf.nn.relu(bn(im, is_t=is_t, name='conv_mf_bn1_{}'.format(i)))
                im = slim.separable_conv2d(im, 728, [3,3], scope='conv_mf_sp2_{}'.format(i))
                im = tf.nn.relu(bn(im, is_t=is_t, name='conv_mf_bn2_{}'.format(i)))
                im = slim.separable_conv2d(im, 728, [3,3], scope='conv_mf_sp3_{}'.format(i))
                im = bn(im, is_t=is_t, name='conv_mf_bn3_{}'.format(i))
                if senet == True:
                    im = se_layer(im, 728, 8, name='conv_se_728_{}'.format(i))
                im = tf.add(im, im_res, name='con_mf_add_{}'.format(i))
                
            #Exit flow
            im_res = conv2d(im, output_dim=1024, k_w=1, k_h=1,stride=2, name='conv_res_ex_1024')
            im_res = bn(im_res, is_t=is_t, name='conv_res_ex_bn_1024')
            im = tf.nn.relu(im, name='conv_exit_relu')
            im = slim.separable_conv2d(im, 728, [3,3], scope='conv_ex_728')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_ex_bn_728'))
            im = slim.separable_conv2d(im, 1024, [3,3], scope='conv_ex1_1024')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_ex1_bn_1024'))
            im = slim.separable_conv2d(im, 1024, [3,3], stride=2, scope='conv_ex2_1024')
            im = bn(im, is_t=is_t, name='conv_ex2_bn_1024')
            if senet == True:
                im = se_layer(im, 1024, 8, name='conv_se_1024')
            im = tf.add(im, im_res, name='conv5_add')
            #Output
            im = tf.nn.relu(im)
            im = slim.separable_conv2d(im, 1536, [3,3], scope='conv_out_1536')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_out_bn_1536'))
            im = slim.separable_conv2d(im, 1536, [3,3], scope='conv_out2_1536')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_out2_bn_1536'))
            im = slim.separable_conv2d(im, 2048, [3,3], scope='conv_out_2048')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_out_bn_2048'))                  
            if senet == True:
                im = se_layer(im, 2048, 8, name='conv_se_2048')            
            return im, skip   

def decoder_block(img, skip, is_t):
    with tf.variable_scope('gen_upsample'):
        with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1):
            shape = tf.shape(img)
            h,w = shape[1], shape[2]
            #block0 output_size=16
            im = tf.image.resize_bilinear(img, [h*2,w*2], name='upsample_16')
            im = slim.separable_conv2d(im, 512, [3,3], scope='conv_sp_512')
            im = bn(im, is_t=is_t, name='bn_sp_512')
            im = concat(im, skip[4], name='cat_512')                  
            #block1 output_size=32
            im = conv2d(im, output_dim=256, name='conv_256')
            im = tf.image.resize_bilinear(im, [h*4,w*4], name='upsample_32')
            im = slim.separable_conv2d(im, 256, [3,3], scope='conv_sp_256')
            im = bn(im, is_t=is_t, name='bn_sp_256')
            im = concat(im, skip[3], name='cat_256')   
            #block2 output_size=64
            im = conv2d(im, output_dim=128, name='conv_128')
            im = tf.image.resize_bilinear(im, [h*8,w*8], name='upsample_64')
            im = slim.separable_conv2d(im, 128, [3,3], scope='conv_sp_128')
            im = bn(im, is_t=is_t, name='bn_sp_128')
            im = concat(im, skip[2], name='cat_128')
            #block3 output_size=128
            im = conv2d(im, output_dim=64, name='conv_64')
            im = tf.image.resize_bilinear(im, [h*16,w*16], name='upsample_32')
            im = slim.separable_conv2d(im, 64, [3,3], scope='conv_sp_64')
            im = bn(im, is_t=is_t, name='bn_sp_64')
            im = concat(im, skip[1], name='cat_64')  
            #block2 output_size=256
            im = conv2d(im, output_dim=32, name='conv_32')
            im = tf.image.resize_bilinear(im, [h*32,w*32], name='upsample_128')
            im = slim.separable_conv2d(im, 32, [3,3], scope='conv_sp_32')
            im = bn(im, is_t=is_t, name='bn_sp_32')
            im = concat(im, skip[0], name='cat_32') 
            #output
            im = conv2d(im, output_dim=16, name='output_16')
            im = tf.nn.relu(bn(im, is_t=is_t, name='output_bn_16'))
            im = conv2d(im, output_dim=3, k_h=1, k_w=1, name='output_3')
        
            return im
    
def disc_block(img, is_t, batch_size, sn, uc, senet):
    with tf.variable_scope('dis'):
        #block0 output_size=128
        im = conv2d(img, output_dim=32, stride=2, spectral_normed=sn, update_collection=uc, name='conv_1_32')
        im = lrelu(im)
        im = conv2d(im, output_dim=32, k_h=1, k_w=1, spectral_normed=sn, update_collection=uc, name='conv_2_32')
        if senet == True:
            im = se_layer(im, 32, 8, name='conv_se_32')
        #block1 output_size=64
        im = conv2d(im, output_dim=64, stride=2, spectral_normed=sn, update_collection=uc, name='conv_1_64')
        im = lrelu(im)
        im = conv2d(im, output_dim=64, k_h=1, k_w=1, spectral_normed=sn, update_collection=uc, name='conv_2_64')
        if senet == True:
            im = se_layer(im, 64, 8, name='conv_se_64')
        #block2 output_size=32
        im = conv2d(im, output_dim=128, stride=2, spectral_normed=sn, update_collection=uc, name='conv_1_128')
        im = lrelu(im)
        im = conv2d(im, output_dim=128, k_h=1, k_w=1, spectral_normed=sn, update_collection=uc, name='conv_2_128')
        if senet == True:
            im = se_layer(im, 128, 8, name='conv_se_128')
        #block3 output_size=16
        im = conv2d(im, output_dim=256, stride=2, spectral_normed=sn, update_collection=uc, name='conv_1_256')
        im = lrelu(im)
        im = conv2d(im, output_dim=256, k_h=1, k_w=1, spectral_normed=sn, update_collection=uc, name='conv_2_256')
        if senet == True:
            im = se_layer(im, 256, 8, name='conv_se_256')
        #block4 output_size=8
        im = conv2d(im, output_dim=512, stride=2, spectral_normed=sn, update_collection=uc, name='conv_1_512')
        im = lrelu(im)
        im = conv2d(im, output_dim=512, k_h=1, k_w=1, spectral_normed=sn, update_collection=uc, name='conv_2_512')
        if senet == True:
            im = se_layer(im, 512, 8, name='conv_se_512')
         
        return im    

    
def atrous_spatial_pyramid_pooling(net, scope="aspp", depth=256):

    with tf.variable_scope(scope):
        feature_map_size = tf.shape(net)
        # apply global average pooling
        image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keepdims=True)
        image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1", activation_fn=None)
        image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))
        at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)
        at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=6, activation_fn=None)
        at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=12, activation_fn=None)
        at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=18, activation_fn=None)
        net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3, name="concat")
        net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)
        return net

        
        
        