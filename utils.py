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
        img = np.resize(img, [256, 256, 1])
        
    img = np.array(img).astype(np.float)
    img = img/127.5 - 1.
    
    
    return img


def save_data(images):
    img = (images + 1) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img




