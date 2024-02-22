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
@click.option('-i','--image_dir',default='./3_Test1_Image/',help='image directory')
def test_image(image_dir):
    '''read .nii.gz file and output jpg files into directory'''
    nii_filelist = glob(image_dir+'*.nii.gz')
    #print(mask_filelist)

    label_array = []
    
    for ix, nii_file in enumerate(tqdm(nii_filelist)):
        filename = os.path.basename(nii_file) 
        img = nib.load(nii_file)
        if isinstance(img, type(None)):
            print('File {} does not include nii image'.format(nii_file))
            continue
        img_data = img.get_fdata()
        slice_num = img_data.shape[2]
        for num in range(slice_num):
            clip_array = np.clip(img_data[:, :, num], -150, 250)
            clip_array = (clip_array - (-150)) / (250 - (-150))
            clip_array = clip_array * 255
            
            img_out_path = image_dir+'jpg/'+filename.replace(".nii.gz",'')

                
            img_out_fname = '{}_{}.jpg'.format(img_out_path,num)

            cv2.imwrite(img_out_fname, clip_array)


    return





if __name__=='__main__':
    test_image()
