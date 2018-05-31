import os
import sys
import time
import numpy as np

import scipy.misc
import skimage

import utils
import model as modellib
from config import InferenceConfig

ROOT_DIR = os.getcwd()

# SET THIS UP:
MODEL_PATH = "./weights/ch1_ch2_new100_150epochs_coco_berkeley.h5"
IMAGE_DIR  = "./inputfiles/"
SAVE_DIR   = "./outfiles/"

def main():
    inference_config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_PATH)
    model.load_weights(MODEL_PATH, by_name=True)

    file_names = next(os.walk(IMAGE_DIR))[2]

    for fn in file_names:
        print(fn)
        image = skimage.io.imread(os.path.join(IMAGE_DIR, fn))
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
        scipy.misc.imsave(SAVE_DIR + fn, MI)
    print('done')

if __name__ == "__main__":
    main()