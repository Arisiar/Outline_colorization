import os
from glob import glob
import tensorflow as tf
import numpy as np
from utils import load_data, save_data
from network.U_net import U_net
from network.Res_U_net import Res_U_net 
from network.Deeplabv3_Plus import DeeplabV3_Plus
import cv2
slim = tf.contrib.slim
class Colorization(object):
    
    
    def __init__(self, in_h, in_w, senet, batch_size, learn_rate, dataset_name, checkpoint_dir, sample_dir, network):
        

        # model format
        self.batch_size = batch_size  
        self.senet = senet
        self.learning_rate = learn_rate
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.network = network
        # placeholder
        self.outlines_data = tf.placeholder(tf.float32, [self.batch_size, in_h, in_w, 1],name='outlines_images')
        self.real_data = tf.placeholder(tf.float32, [self.batch_size, in_h, in_w, 3],name='color_images') 
        self.sample = tf.placeholder(tf.float32, [self.batch_size, in_h, in_w, 1],name='sample_images') 
        # record data
        self.log_vars = []
        self.fname = []    
        
    def init_opt(self):
         
        #for generator
        self.fake_images = self.model(self.outlines_data, ntype='gen', is_t=True, uc="NO_OPS", senet=self.senet, reuse=False)
        self.sample_images = self.model(self.sample, ntype='gen', is_t=False, uc="NO_OPS", senet=self.senet, reuse=True)
        #for discriminator    
        G_fake = self.model(self.fake_images, ntype='dis', is_t=True, uc=None, senet=self.senet, reuse=False)
        D_real = self.model(self.real_data, ntype='dis', is_t=True, uc="NO_OPS", senet=self.senet, reuse=True)
        #compute loss
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
        G_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=G_fake, labels=tf.zeros_like(G_fake)))
        G_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=G_fake, labels=tf.ones_like(G_fake)))
        
        self.L1 = tf.reduce_mean(tf.abs(self.fake_images - self.real_data))
        self.D_loss = D_loss_real + G_loss_fake
        self.G_loss = G_loss_real  + self.L1
        
        # summary 
        tf.summary.scalar("G_loss", self.G_loss)
        tf.summary.scalar("D_loss", self.D_loss)
        tf.summary.scalar("L1_loss", self.L1)
        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))
        self.log_vars.append(("L1_loss", self.L1))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        
        self.saver = tf.train.Saver()
          
    def train(self):
         
        d_optim = tf.train.AdamOptimizer(self.learning_rate) \
                          .minimize(self.D_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.learning_rate) \
                          .minimize(self.G_loss, var_list=self.g_vars)
        
        images = glob('./datasets/{}/train/images/*.jpg'.format(self.dataset_name))
        outlines = glob('./datasets/{}/train/outlines/*.jpg'.format(self.dataset_name))
        batch_idxs = int(len(images)/self.batch_size) 
         
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        num = 0
        with tf.Session(config=config) as sess:
            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter("./logs", sess.graph)   
            print(" [*] Use the {} model for training...".format(self.network))
            print(" [*] Reading checkpoint...")
            for epoch in range(40): 
                  counter = 0 
                  # LOad Model
                  ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                  if ckpt and ckpt.model_checkpoint_path:
                      print(" [*] Model Load Success...")
                      self.saver.restore(sess, os.path.join(self.checkpoint_dir, "./Colorization"))
                  else:
                      print(" [!] Model Load Failed...")
                  #Load Data
                  for idx_batch in range(batch_idxs):          
                      #data about realimage
                      batch_images_files = images[counter*self.batch_size:(counter+1)*self.batch_size] 
                      batch_images = np.array([load_data(batch_file) for batch_file in batch_images_files]).astype(np.float32)

                      # data about outlines
                      batch_outlines_files = outlines[counter*self.batch_size:(counter+1)*self.batch_size]
                      batch_outlines = np.array([load_data(batch_file, n='gray') for batch_file in batch_outlines_files]).astype(np.float32)
                    
                      #Optimization D
                      sess.run(d_optim, feed_dict={self.real_data: batch_images, self.outlines_data: batch_outlines})
                      # Optimization G 
                      sess.run(g_optim, feed_dict={self.real_data: batch_images, self.outlines_data: batch_outlines})
                
                      summary_str = sess.run(summary_op, feed_dict={self.real_data: batch_images, self.outlines_data: batch_outlines})
                      summary_writer.add_summary(summary_str, counter)
                 
                      D_loss, G_loss, L1 = sess.run([self.D_loss, self.G_loss, self.L1], feed_dict={self.real_data: batch_images, self.outlines_data: batch_outlines})
                      print("Epoch[%2d]:[%4d/%4d] D_loss: %.7f, G_loss: %.7f, L1 %.7f"% (epoch, counter, batch_idxs, D_loss, G_loss, L1))  
                      
                      if np.mod(counter, 100) == 0:
                          self.sample_model(self.sample_dir, num, sess)
                          num += 1
                      if np.mod(counter, 2000) == 0:
                          self.saver.save(sess,os.path.join(self.checkpoint_dir, "./Colorization"))
                      counter += 1
                  self.saver.save(sess,os.path.join(self.checkpoint_dir, "./Colorization"))
            sess.close()

    def model(self, img, ntype, is_t=True, uc=None, senet=False, reuse=False):
 
        
        if self.network == "U_net":
            net = U_net(img, is_t, self.batch_size, uc, senet)
        elif self.network == "Res_U_net":
            net = Res_U_net(img, is_t, self.batch_size, uc, senet)
        elif self.network == "DeeplabV3_Plus":
            net = DeeplabV3_Plus(img, is_t, self.batch_size, uc, senet)        
        
        if ntype == 'dis':
            with tf.variable_scope("discriminator") as scope:
                if reuse == True:
                    scope.reuse_variables()
                net =net.discriminator()
        elif ntype == 'gen':
            with tf.variable_scope("generator") as scope:
                if reuse == True:
                    scope.reuse_variables()
                net = net.generator()
        
        return net

    def load_sample(self,num):
        
        data = glob('./datasets/outlines/val/outlines/*.jpg')
        fname = []
        img = data[num*self.batch_size:(num+1)*self.batch_size]
        for i in range(self.batch_size):
            fname.append(str(img[i]).split("\\")[-1].split('.')[0])
        sample = np.array([load_data(sample_file, n='gray') for sample_file in img]).astype(np.float32)
        
        return sample, fname
            
    def sample_model(self, sample_dir, num, sess):

        sample_images, fname = self.load_sample(num)
        sample = sess.run(self.sample_images, feed_dict={self.sample: sample_images})  
        sample = save_data(sample)
        for i in range(self.batch_size):
            cv2.imwrite("./sample/{}.jpg".format(i), sample[i])

    def test(self, sess):
        
        self.saver.restore(sess, os.path.join(self.checkpoint_dir, "./Colorization"))
        data = glob('./test/*.jpg')
        img = data[0:self.batch_size]
        sample = np.array([load_data(sample_file, n='gray') for sample_file in img]).astype(np.float32)
        sample = sess.run(self.sample_images, feed_dict={self.sample: sample})
        sample = save_data(sample)
        for i in range(self.batch_size):
            cv2.imwrite('./result/{}.jpg'.format(i), sample[i])        
        
        
        
        
        