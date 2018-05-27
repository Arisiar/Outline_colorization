from __future__ import division
import cv2
import numpy as np

def load_data(image_path, n='default'):
    
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256), interpolation= cv2.INTER_AREA)
    if n == 'blur':
        img = cv2.blur(img, (3,3))     
    elif n == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.fastNlMeansDenoising(img,3,3,7,21)
        img = np.resize(img, [256, 256, 1])
        
    img = np.array(img).astype(np.float)
    img = img/127.5 - 1.
    
    
    return img


def save_data(images):
    img = (images + 1) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return  cv2.imwrite(path, merge(images, size))


def inverse_transform(images):
    return (images+1.)/2.


