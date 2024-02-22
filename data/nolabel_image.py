#!/usr/bin/env python3.9
# 

import os
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import pickle as pickle
import cv2
from tqdm import tqdm
import click
from glob import glob

# program to convert a  set into a list
def convert(s):
    return list(map(lambda x: x, s))



@click.command()
@click.option('-i','--image_dir',default='./1_Train,Valid_Image/',help='image directory')
@click.option('-l','--label_dir',default='./2_Train,Valid_Mask/' ,help='label directory')
def nolabel_image(image_dir,label_dir):
    '''read label files and write anatation of segmentation'''
    mask_filelist = glob(label_dir+'*.nii.gz')
    #print(mask_filelist)

    label_array = []
    
    for ix, label_file in enumerate(tqdm(mask_filelist)):
 
        n1_img = nib.load(label_file)
        label_data = n1_img.get_fdata()
        label_list = (np.where(np.any(n1_img.get_fdata()==1, axis=0))[1])
        
        islabeled_list = sorted(convert(set(label_list.flatten())))
        if len(islabeled_list)>0:
            continue
        tot_picnum = n1_img.shape[2]
        nolabeled_list = [x for x in range(tot_picnum) if x not in islabeled_list]
        filename = os.path.basename(label_file)

        img_fname = image_dir+filename.replace("_label.nii.gz",'.nii.gz')
        img = nib.load(img_fname)
        if isinstance(img, type(None)):
            print('File {} does not include nii image'.format(img_fname))
            continue
        img_data = img.get_fdata()

        for num in nolabeled_list:
            label_img = n1_img.get_fdata()[:, :, num]
            image_size = [label_img.shape[1],label_img.shape[0]]
            
                            
            clip_array = np.clip(img_data[:, :, num], -150, 250)
            clip_array = (clip_array - (-150)) / (250 - (-150))
            clip_array = clip_array * 255
            
            img_out_path = image_dir+'nolabel/'+filename.replace("_label.nii.gz",'')
                
            img_out_fname = '{}_{}.jpg'.format(img_out_path,num)

            cv2.imwrite(img_out_fname, clip_array)


    return





if __name__=='__main__':
    nolabel_image()
