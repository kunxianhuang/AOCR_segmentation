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

def find_contours(slice_img):
    imgray = slice_img*255.0
    ret, thresh = cv2.threshold(imgray, 127, 255,cv2.THRESH_BINARY)
    thresh = np.clip(thresh, 0,255)
    thresh = np.array(thresh,np.uint8) 
    contour,hi = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contour


@click.command()
@click.option('-l','--label_dir',default='./2_Train,Valid_Mask/' ,help='label directory')
@click.option('-i','--image_dir',default='./1_Train,Valid_Image/',help='image directory')
def ananotation(label_dir,image_dir):
    '''read label files and write anatation of segmentation'''
    mask_filelist = glob(label_dir+'*.nii.gz')
    #print(mask_filelist)

    label_array = []
    
    for ix, label_file in enumerate(tqdm(mask_filelist)):
 
        n1_img = nib.load(label_file)
        label_data = n1_img.get_fdata()
        havelabel = (np.where(np.any(n1_img.get_fdata()==1, axis=0))[1])
        islabeled_list = sorted(convert(set(havelabel.flatten())))
        filename = os.path.basename(label_file)
        
        img_fname = image_dir+filename.replace("_label.nii.gz",'.nii.gz')
        img = nib.load(img_fname)
        if isinstance(img, type(None)):
            print('File {} does not include nii image'.format(img_fname))
            continue
        img_data = img.get_fdata()
        
        for num in islabeled_list:
            label_img = n1_img.get_fdata()[:, :, num]
            image_size = [label_img.shape[1],label_img.shape[0]]
            contours = find_contours(label_img)
            if len(contours)>0:
                # save label contour txt files
                if ix <50:
                    # validation set of 50
                    label_txt_dir =  label_dir+'seganon_txt_val/'
                    if not os.path.exists(label_txt_dir): 
                        # if the demo_folder directory is not present  
                        # then create it. 
                        os.makedirs(label_txt_dir) 
                else:
                    label_txt_dir =  label_dir+'seganon_txt/'
                    if not os.path.exists(label_txt_dir): 
                        # if the demo_folder directory is not present  
                        # then create it. 
                        os.makedirs(label_txt_dir)
                        
                label_txt_fname_ = label_txt_dir+filename.replace("_label.nii.gz",'')
                label_txt_fname = '{}_{}.txt'.format(label_txt_fname_,num)
                with open(label_txt_fname,'w+') as f_label:
                    for contour in contours:
                        contour = np.flip(contour,0)
                        label_array.append({(filename.replace("_label.nii.gz",''),num) : contour})
                        #label_for_file.append({'path':(label_paths.replace("_label.nii.gz",''),num),'label' : contour})
                        #label_dict[(filename.replace("_label.nii.gz",''),num)] = contour
                    
                        # label of appendicitis
                        label_appendicitis = '1 '
                        contour_txt = ''
                        contour_txt = contour_txt + label_appendicitis
                        for points in contour:
                            [x,y] = points[0]
                            x = x*1.0/label_img.shape[0]
                            y = y*1.0/label_img.shape[1]
                            x_y_pt = '{} {} '.format(x,y)
                            contour_txt = contour_txt + x_y_pt

                        contour_txt = contour_txt+'\n'
                        f_label.write(contour_txt)
                        
                
            clip_array = np.clip(img_data[:, :, num], -150, 250)
            clip_array = (clip_array - (-150)) / (250 - (-150))
            clip_array = clip_array * 255
            if ix <50:
                # validation set of 50
                img_out_path = image_dir+'jpg_val/'+filename.replace("_label.nii.gz",'')
            else:
                img_out_path = image_dir+'jpg/'+filename.replace("_label.nii.gz",'')
                
            img_out_fname = '{}_{}.jpg'.format(img_out_path,num)

            cv2.imwrite(img_out_fname, clip_array)

            
            
    return

                
if __name__=='__main__':
    ananotation()
