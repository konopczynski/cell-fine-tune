# Training and validation functions for 2D/3D Liver/Lesion Segmentation Training
# of neural networks using the PyTorch framework.
# @author: Karsten Roth - Heidelberg University, 07/11/2017

"""==============================================================="""
### LIBRARIES
import warnings
warnings.filterwarnings("ignore")
import torch, random, numpy as np, sys, os, time
sys.path.insert(0, os.getcwd()+'/../Utilities')
import General_Utilities as gu, Network_Utilities as nu
from tqdm import tqdm, trange
import matplotlib.pyplot as plt



def unpad_img(img, orig_x, orig_y):
    x0 = 16
    y0 = 16
    x1 = 16
    y1 = 16
    if (orig_y % 32) != 0:
        x0 = int((32 - orig_y % 32) / 2)
        x1 = (32 - orig_y % 32) - x0
        x0 += 16
        x1 += 16
    if (orig_x % 32) != 0:
        y0 = int((32 - orig_x % 32) / 2)
        y1 = (32 - orig_x % 32) - y0
        y0 += 16
        y1 += 16
    img0 = img[y0:-y1, x0:-x1] 
    return img0

"""==============================================================="""
### VALIDATE ANY 2D SEGMENTATION NETWORK FOR LIVER/LESION SEGMENTATION
def validator(network, data_loader, opt, folder_name='predictions'):
    os.makedirs(folder_name, exist_ok=True)
    _ = network.eval()

    validation_data_iter = tqdm(data_loader, position=2)
    
    for slice_idx, file_dict in enumerate(validation_data_iter):
        validation_slice = file_dict["image"].type(torch.FloatTensor).to(opt.device)
        network_output   = network(validation_slice)[0][0].cpu().detach().numpy()[0]
        network_output   = unpad_img(network_output, file_dict["shape"][0].item(), file_dict["shape"][1].item())
        plt.imsave(os.path.join('..', folder_name, file_dict['name'][0][:-3]+'png'), network_output)
