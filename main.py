import argparse
import os
import tensorflow as tf
from model import Colorization

parser = argparse.ArgumentParser(description='')
parser.add_argument('--datasets', dest='datasets', default='outlines', help='name of the dataset')
parser.add_argument('--in_h', dest='in_h', default=256, help='height of the input')
parser.add_argument('--in_w', dest='in_w', default=256, help='width of the input')
parser.add_argument('--senet', dest='senet', type=bool, default=True, help='set senet structure')
parser.add_argument('--epoch', dest='epoch', type=int, default=20, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='# images in batch')                   
parser.add_argument('--learn_rate', dest='learn_rate', type=float, default=0.0005, help='initial learning rate for adam')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--result_dir', dest='result_dir', default='./test', help='test result are saved here')
parser.add_argument('--training', dest='training', type=bool, default=True, help='test sample are saved here')
parser.add_argument('--network', dest='network', default='DeeplabV3_Plus', help='Differ model which you want to load(U-net | Res_U_net | DeeplabV3_Plus) ')
args = parser.parse_args()


if __name__ == "__main__":
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = Colorization(in_h=args.in_h, in_w=args.in_w, senet=args.senet, batch_size=args.batch_size, 
                         learn_rate=args.learn_rate, dataset_name=args.datasets, network=args.network,
                         checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir)
    model.init_opt()
    
    if args.training:
        model.train()
    else:
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            model.test(sess)        
                   
        



    
    
