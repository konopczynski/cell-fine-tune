# Based on: Expandable script for Lesion Segmentation Base_Unet_Template.py.
# This Variant: Liver-Segmentation
# @author: Karsten Roth - Heidelberg University, 07/11/2017
"""==================================================================================================="""
import shutil
import warnings
warnings.filterwarnings("ignore")
import os,sys, imp
from tqdm import tqdm
import torch
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
#os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT_DIR+'/../Utilities')
sys.path.insert(0, ROOT_DIR+'/../Network_Zoo')
sys.path.insert(0, ROOT_DIR+'/../predict')
import numpy as np

import network_zoo as netlib
import General_Utilities as gu
import Network_Utilities as nu

import PyTorch_Datasets as Data
import Test_Dataset as Cdata
import Function_Library_Test as flib
import model as modellib

SAVE_DIR = os.path.join(ROOT_DIR, '../karsten_outfiles')


def infer(opt):
    
    """======================================================================================="""
    ### GPU SETUP
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.Training['gpu'])
    opt.device = torch.device('cpu')

    ### LOSS SETUP
    base_loss_func      = nu.Loss_Provider(opt)
    aux_loss_func       = nu.Loss_Provider(opt) if opt.Network['use_auxiliary_inputs'] else None
    opt.Training['use_weightmaps']  = base_loss_func.loss_func.require_weightmaps
    opt.Training['require_one_hot'] = base_loss_func.loss_func.require_one_hot
    opt.Training['num_out_classes'] = 1 if base_loss_func.loss_func.require_single_channel_input else opt.Training['num_classes']

    ### NETWORK SETUP
    network = netlib.NetworkSelect(opt)
            
    network.n_params = nu.gimme_params(network)
    opt.Network['Network_name'] = network.name
    _ = network.to(opt.device)
    check_path = os.path.join(ROOT_DIR, '../../weights/best_karsten_model.pth.tar')  
    checkpoint = torch.load(check_path)
    network.load_state_dict(checkpoint['network_state_dict'])

    """======================================================================================="""
    imp.reload(Data)
    test_dataset = Cdata.Generate_Datasets(opt)#Data.Generate_Required_Datasets(opt)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=1, pin_memory=False, shuffle=False)
    ###### Validation #########
    flib.validator(network, test_data_loader, opt, folder_name=SAVE_DIR)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    

def run_infer():
    opt = Namespace(base_setup = 'Baseline_Parameters.txt', search_setup = 'Small_UNet_Liver.txt')

    opt.base_setup   = ROOT_DIR+'/Training_Setup_Files/' + opt.base_setup
    opt.search_setup = ROOT_DIR+'/Training_Setup_Files/' + opt.search_setup

    training_setups = gu.extract_setup_info(opt)
    for training_setup in tqdm(training_setups, desc='Setup Iteration... ', position=0):
        infer(training_setup)
    #os.chdir(ROOT_DIR)
