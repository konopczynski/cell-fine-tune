import numpy as np
from skimage.measure import regionprops
import os
import cv2
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt

def describe(df):
    return pd.concat([df.describe().T,
                      df.mad().rename('mad'),
                      df.skew().rename('skew'),
                      df.kurt().rename('kurt'),
                     ], axis=1).dropna().T

def separate_masks(mask):
    r = []
    idxs = []
    m = np.max(mask)+1
    for i in range(1, m):
        if(i in mask):
            r.append((mask == i))
            idxs.append(i)
    r = np.swapaxes(r, 0, 2)
    r = np.swapaxes(r, 0, 1)
    return r, idxs

def calc_area_perim(masks):
    perims = [regionprops(masks[:, :, i].astype(int))[0].perimeter for i in range(masks.shape[-1])]
    masks_flatten = np.reshape(masks > .5, (-1, masks.shape[-1])).astype(np.float32)
    areas = np.sum(masks_flatten, axis=0)
    return areas, perims


class TestDataset():
    def __init__(self):
        self.mask_info = {}
        self.image_ids = []
    
    def load_pred(self, pred_path):
        for file in os.listdir(pred_path):
            image_id = file[:-4]
            self.image_ids.append(image_id)
            pred_mask_path = os.path.join(pred_path, file)
            pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
            
            self.mask_info[image_id] = pred_mask
            
    def get_cells(self, iou_threshold = 0.5):
        
        if not self.mask_info:
            print('Load masks first with .load_masks()')
        else:
            total_area = []
            total_perim = []
            total_ids = []
            for id_ in self.image_ids:
                # Load prediction
                pred_mask = self.mask_info[id_]
                # Separate cells from one image
                pred_mask, pred_idxs = separate_masks(pred_mask)
                #get lists with areas and perimeters
                area, perim = calc_area_perim(pred_mask)
                total_area += list(area)
                total_perim += list(perim)
                total_ids += [id_ for i in range(len(area))]
            total_asphericity = total_area / np.square(total_perim)
            self.cell_info = pd.DataFrame({'area': total_area, 'perimeter': total_perim, 'asphericity': total_asphericity,
                                           'image_id': total_ids})

    def get_statistics(self, bins=20, logscale=False):
        if self.cell_info is None:
            print('Get cell data first with .get_cells()')        
        else:
            df = self.cell_info.copy()
            
            print('Dataset has in total ', len(self.cell_info), 'cells')
            fig, ax = plt.subplots(nrows=3, figsize=(15, 25))
            
            for i, feature in enumerate(['area', 'asphericity', 'perimeter']):
                
                df.sort_values(by=feature, inplace=True)
                total = df[feature].tolist()
                x_title = feature

                n_total, bin_vals, _ = ax[i].hist(total, bins=bins, log=logscale)
                n_total = [x+0.00001 for x in n_total]
                ax[i].set_title('Distribution of ' + feature + ' for predicted cells')
                ax[i].set_xlabel(x_title)
                
                print('For ' + feature + ':')
                for num in range(bins):
                    print('In range:', np.round(bin_vals[num], 3), ",", np.round(bin_vals[num+1], 3), 'total cells: ', int(n_total[num]))