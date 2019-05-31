import os

from tqdm import tqdm
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import warnings
warnings.filterwarnings("ignore")

import scipy.misc
import skimage

import utils
import model as modellib
from config import InferenceConfig

from skimage import img_as_uint
import unet.predict.infer_unet as unet

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def main(opt):
    inference_config = InferenceConfig()
    image_dir = opt.image_folder
    save_dir = opt.save_folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if opt.model == 'hybrid':
        feature_dir = 'unet/unet_outfiles'
        model_path = 'weights/MaskRCNN_hybrid.h5'
        
        os.makedirs(feature_dir, exist_ok=True)
        unet.run_infer(image_dir)

        model = modellib.MaskRCNN(mode="inference", 
                                  config=inference_config,
                                  model_dir=model_path)
        model.load_weights(model_path, by_name=True)

        file_names = next(os.walk(image_dir))[2]
        for fn in tqdm(file_names):
            img = plt.imread(os.path.join(image_dir, fn))
            feature = rgb2gray(plt.imread(os.path.join(feature_dir, fn[:-3]+'png')))
            combined = np.zeros((img.shape[0], img.shape[1], 3))
            combined[:, :, 0] = img
            combined[:, :, 1] = img
            combined[:, :, 2] = feature
            original_image = combined
            results = model.detect([original_image], verbose=0)
            r = results[0]
            RM =  r['masks']
            MI = np.zeros(shape=original_image[:, :, 0].shape)
            for i in range(np.shape(RM)[2]):
                MI[RM[:,:,i]==1] = i
            scipy.misc.imsave(os.path.join(save_dir, fn[:-3]+'png'), MI)
            
    elif opt.model == 'pure':
        model_path = 'weights/MaskRCNN_pure.h5'
        model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=model_path)
        model.load_weights(model_path, by_name=True)

        file_names = next(os.walk(image_dir))[2]
        for fn in tqdm(file_names):
            image = skimage.io.imread(os.path.join(image_dir, fn))
            z = np.zeros(shape=(np.shape(image)[0],np.shape(image)[1],3))
            z[:,:,0]=image
            z[:,:,1]=image
            z[:,:,2]=image
            original_image=z

            results = model.detect([original_image], verbose=0)
            r = results[0]
            RM =  r['masks']
            NI = original_image[:,:,0].copy()
            MI = np.zeros(shape=np.shape(NI))

            for i in range(np.shape(RM)[2]):
                NI[RM[:,:,i]==1] = 255
                MI[RM[:,:,i]==1] = i
            scipy.misc.imsave(os.path.join(save_dir, fn[:-3]+'png'), MI)
    print('done')

if __name__ == "__main__":
    
    parse_in = argparse.ArgumentParser()
    parse_in.add_argument('--model',   type=str, default='hybrid', choices = ['hybrid', 'pure'],
                                            help='Choose model for prediction')
    parse_in.add_argument('--image_folder', type=str, default="./inputfiles/", help='Directory with test images')
    parse_in.add_argument('--save_folder', type=str, default="./outfiles/", help='Directory to save predicted masks')
    opt = parse_in.parse_args()

    main(opt)
