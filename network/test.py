import tensorflow as tf
from ops import linear, conv2d, lrelu, bn, concat, se_layer, pixel_dcl
slim = tf.contrib.slim
#%%
class test(object):

    def __init__(self, img, is_t, batch_size, uc, senet):
        
        self.img = img
        self.is_t = is_t
        self.batch_size=batch_size
        self.uc = uc        
        self.senet = senet
    
    def discriminator(self):

        with tf.variable_scope('dis'):
         
            net = disc_block(self.img, self.is_t, self.batch_size, True, self.uc, self.senet)
            net = linear(tf.reshape(net, [self.batch_size, -1]), 1, spectral_normed=True, update_collection=self.uc, scope='linear')    
           
            return net 

    def generator(self):
    
        with tf.variable_scope("gen"):
                
            net, fe = encoder_block(self.img, self.is_t, self.senet)      
            net = decoder_block(net, fe, self.is_t)

            return tf.nn.tanh(net) 

#%% 
def encoder_block(img, is_t, senet):
        
    with tf.variable_scope('downsample_space'):
        with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1):
               
            feature = []
            
            
            im = conv2d(img, 16, k_w=1, k_h=1, name='conv_16')
            im = bn(im, is_t=is_t, name='conv_bn_16')
            feature.append(im)
            # block1 output_size=128
            im = slim.separable_conv2d(im, 32, [3,3], scope='conv_0_32')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_0_bn_32'))
            im = slim.separable_conv2d(im, 32, [3,3], scope='conv_1_32')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_1_bn_32'))        
            im_fe = slim.max_pool2d(im, [3,3], stride=2, padding='same', scope='conv_max_pool')
            im_sp = slim.separable_conv2d(im, 32, [3,3], stride=2, scope='conv_sp_32')
            im = tf.concat([im_sp, im_fe], 3)
            im = conv2d(im, 32, k_h=1, k_w=1, name='conv_cat_32')
            im = bn(im, is_t=is_t, name='conv_cat_bn_32')
            im_res = slim.separable_conv2d(im, 64, [3,3], stride=2, scope='conv_res_64')
            im_res = bn(im_res, is_t=is_t, name='conv_res_bn_64')
            feature.append(im)
            
            # block2 output_size=64
            im = slim.separable_conv2d(im, 64, [3,3], scope='conv_0_64')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_0_bn_64'))
            im = slim.separable_conv2d(im, 64, [3,3], scope='conv_1_64')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_1_bn_64'))  
            im_fe = slim.max_pool2d(im, [3,3], stride=2, padding='same', scope='conv_max_pool')
            im_sp = slim.separable_conv2d(im, 64, [3,3], stride=2, scope='conv_sp_64')
            im = tf.concat([im_sp, im_fe], 3)
            im = conv2d(im, 64, k_h=1, k_w=1, name='conv_cat_64')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_cat_bn_64') + im_res)
            im = se_layer(im, 64, 16, name='conv_se_64')
            im_res = conv2d(im, 128, k_h=1, k_w=1, stride=2, name='conv_res_128')
            im_res = bn(im_res, is_t=is_t, name='conv_res_bn_128')
            feature.append(im)
           
            # block3 output_size=32  
            im = slim.separable_conv2d(im, 128, [3,3], scope='conv_0_128')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_0_bn_128'))
            im = slim.separable_conv2d(im, 128, [3,3], scope='conv_1_128')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_1_bn_128')) 
            im_fe = slim.max_pool2d(im, [3,3], stride=2, padding='same', scope='conv_max_pool')
            im_sp = slim.separable_conv2d(im, 128, [3,3], stride=2, scope='conv_sp_128')
            im = tf.concat([im_sp, im_fe], 3)
            im = conv2d(im, 128, k_h=1, k_w=1, name='conv_cat_128')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_cat_bn_128') + im_res)
            im_res = conv2d(im, 256, k_h=1, k_w=1, stride=2, name='conv_res_256')
            im_res = bn(im_res, is_t=is_t, name='conv_res_bn_256')
            feature.append(im)
            
            # block4 output_size=16
            im = slim.separable_conv2d(im, 256, [3,3], scope='conv_0_256')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_0_bn_256'))
            im = slim.separable_conv2d(im, 256, [3,3], scope='conv_1_256')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_1_bn_256'))        
            im_fe = slim.max_pool2d(im, [3,3], stride=2, padding='same', scope='conv_max_pool')
            im_sp = slim.separable_conv2d(im, 256, [3,3], stride=2, scope='conv_sp_256')
            im = tf.concat([im_sp, im_fe], 3)
            im = conv2d(im, 256, k_h=1, k_w=1, name='conv_cat_256')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_cat_bn_256') + im_res)
            im = se_layer(im, 256, 16, name='conv_se_256')
            im_res = conv2d(im, 512, k_h=1, k_w=1, stride=2, name='conv_res_512')
            im_res = bn(im_res, is_t=is_t, name='conv_res_bn_512')
            feature.append(im)
            
            #block5 output)size=8
            im = slim.separable_conv2d(im, 512, [3,3], scope='conv_0_512')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_0_bn_512'))
            im = slim.separable_conv2d(im, 512, [3,3], scope='conv_1_512')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_1_bn_512'))        
            im_fe = slim.max_pool2d(im, [3,3], stride=2, padding='same', scope='conv_max_pool')
            im_sp = slim.separable_conv2d(im, 512, [3,3], stride=2, scope='conv_sp_512')
            im = tf.concat([im_sp, im_fe], 3)
            im = conv2d(im, 512, k_h=1, k_w=1, name='conv_512')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_bn_512') + im_res)
            im = atrous_spatial_pyramid_pooling(im, scope='conv_aspp_256')
            
            return im, feature


#%%
def decoder_block(img, fe, is_t):
    
    with tf.variable_scope('upsample'):
        with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1):
            
            h,w = int(img.shape[1]), int(img.shape[2])
            
            #block1 output_size=16
            im = pixel_dcl(img, 256, name='pixel_dcl_256')
            im = slim.separable_conv2d(im, 256, [3,3], scope='conv_sp_256')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_bn_sp_256'))
            im = tf.concat([im, fe[4]], 3)
            im = tf.image.resize_bilinear(im, [h*2,w*2], name='upsample_256') 
            im = slim.separable_conv2d(im, 512, [3,3], scope='conv_cat_512')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_bn_cat_512'))   
             
            #block2 output_size=32
            im = pixel_dcl(im, 128, name='pixel_dcl_128')
            im = slim.separable_conv2d(im, 128, [3,3], scope='conv_sp_128')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_bn_sp_128'))
            im = tf.concat([im, fe[3]], 3)
            im = tf.image.resize_bilinear(im, [h*4,w*4], name='upsample_128') 
            im = slim.separable_conv2d(im, 256, [3,3], scope='conv_cat_128')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_bn_cat_128'))  
            im = se_layer(im, 256, 16, name='conv_se_128')
            #block3 output_size=64
            im = pixel_dcl(im, 64, name='pixel_dcl_64')
            im = slim.separable_conv2d(im, 64, [3,3], scope='conv_sp_64')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_bn_sp_64'))
            im = tf.concat([im, fe[2]], 3)
            im = tf.image.resize_bilinear(im, [h*8,w*8], name='upsample_64') 
            im = slim.separable_conv2d(im, 128, [3,3], scope='conv_cat_64')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_bn_cat_64')) 
            
            #block4 output_size=128
            im = pixel_dcl(im, 32, name='pixel_dcl_32')
            im = slim.separable_conv2d(im, 32, [3,3], scope='conv_sp_32')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_bn_sp_32'))
            im = tf.concat([im, fe[1]], 3)
            im = tf.image.resize_bilinear(im, [h*16,w*16], name='upsample_32') 
            im = slim.separable_conv2d(im, 64, [3,3], scope='conv_cat_32')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_bn_cat_32')) 
            im = se_layer(im, 64, 16, name='conv_se_32')
            
            #block4 output_size=256
            im = pixel_dcl(im, 16, name='pixel_dcl_16')
            im = slim.separable_conv2d(im, 16, [3,3], scope='conv_sp_16')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_bn_sp_16'))
            im = tf.concat([im, fe[0]], 3)
            im = tf.image.resize_bilinear(im, [h*32,w*32], name='upsample_16') 
            im = slim.separable_conv2d(im, 32, [3,3], scope='conv_cat_16')
            im = tf.nn.relu(bn(im, is_t=is_t, name='conv_bn_cat_16')) 
            im = conv2d(im, 3, k_w=1, k_h=1, name='conv_output')
            return im
        
#%%
def disc_block(img, is_t, batch_size, sn, uc, senet):
    with tf.variable_scope('dis_disc'):
        with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1):
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
        
#%%  
def atrous_spatial_pyramid_pooling(net, scope="aspp", depth=256):

    with tf.variable_scope(scope):
        feature_map_size = tf.shape(net)
        # apply global average pooling
        image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keep_dims=True)
        image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1", activation_fn=None)
        image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))
        at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)
        at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=3, activation_fn=None)
        at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=6, activation_fn=None)
        at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=9, activation_fn=None)
        net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3, name="concat")
        net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)
        return net
