import os, sys, time, csv, itertools, copy, numpy as np, pandas as pd, itertools as it, torch, copy
import scipy.ndimage.measurements as snm, skimage.transform as st
from tqdm import tqdm, trange
import skimage
import scipy.io
import torchvision
from torchvision.transforms import RandomCrop, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ToTensor, RandomAffine
import random
import PIL
import matplotlib.pyplot as plt


def create_csv(root_dir, save_path):
    names = os.listdir(root_dir)
    df = pd.DataFrame(names, columns = ['id'])
    df.to_csv(save_path)
    return save_path

def Generate_Datasets(opt, test_dir):
    csv_path = create_csv(test_dir, './test_ids.csv')
    test_dataset = CellDataset(csv_path, test_dir, opt, aug=False)
    num_imgs = test_dataset.__len__()
    
    return test_dataset

def pad_img(img):
    x0 = 16
    y0 = 16
    x1 = 16
    y1 = 16
    if (img.shape[1] % 32) != 0:
        x0 = int((32 - img.shape[1] % 32) / 2)
        x1 = (32 - img.shape[1] % 32) - x0
        x0 += 16
        x1 += 16
    if (img.shape[0] % 32) != 0:
        y0 = int((32 - img.shape[0] % 32) / 2)
        y1 = (32 - img.shape[0] % 32) - y0
        y0 += 16
        y1 += 16
    img0 = np.pad(img, ((y0, y1), (x0, x1)), 'edge')
    return img0

class CellDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, opt, aug=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cells_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.pars = opt

        self.transform = torchvision.transforms.ToTensor()

            
    def __len__(self):
        return len(self.cells_frame)

    def __getitem__(self, idx):
        
        img_id = self.cells_frame.iloc[idx, 1]
        img_name = os.path.join(self.root_dir, img_id)
        image = PIL.Image.open(img_name, )
        image = np.array(image)
        image_pad = pad_img(image.copy())
        image_pad = PIL.Image.fromarray(image_pad)
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img transforms 
        if self.transform is not None:
            #image = PIL.Image.fromarray(image)
            image_pad = self.transform(image_pad) 
        
        sample = {'image': image_pad, 'name': img_id, 'shape': (image.shape[0], image.shape[1])}
        
        return sample