#!/usr/bin/env python3.9

import os,sys
from glob import glob
import click
import shutil
from tqdm import tqdm


classp_threshold = 0.6 # threshold of class prediction for false positive samples

@click.command()
@click.option('-l','--label_dir',default='./runs/segment/predict2/labels/' ,help='label directory')
@click.option('-i','--image_dir',default='./data/1_Train,Valid_Image/nolabel/',help='image directory')
@click.option('-i','--save_dir',default='./datasets/aocr-data/',help='training/validation data directory')
def fdata(label_dir,image_dir,save_dir):
    '''read false-positive labels and save its image and label txt to train/val data'''
    mask_filelist = glob(label_dir+'*.txt')
    #print(mask_filelist)                                                                                                

    label_array = []

    for ix, label_file in enumerate(tqdm(mask_filelist)):
        
        filename = os.path.basename(label_file)
        img_fname = image_dir+filename.replace(".txt",'.jpg') # jpg file
        if os.path.isfile(img_fname)==False:
            print("No image {}".format(img_fname))
            continue
        
        with open(label_file,'r') as flabel:
            for fp_seg in  flabel.readlines():
                fp_seg_list = fp_seg.split()
                # remove first element (class)
                fp_seg_list = fp_seg_list[1:]
                classp = float(fp_seg_list[-1])
                # remove last string (classp)
                fp_seg_list = fp_seg_list[:-1]
                # save image and label txt
                if classp >classp_threshold:
                    if ix<2000:
                        save_image_fname = save_dir + "images/val/" + filename.replace(".txt",'.jpg')
                        save_label_fname = save_dir + "labels/val/" + filename
                    else:
                        save_image_fname = save_dir + "images/train/" + filename.replace(".txt",'.jpg')
                        save_label_fname = save_dir + "labels/train/" + filename

                    
                    shutil.copy(img_fname, save_image_fname) # copy image file into training/validation directory
                    # write labels into files with class 0
                    with open(save_label_fname,'a+') as flabel:
                        write_Str ='0 '+' '.join(fp_seg_list)
                        flabel.write(write_Str)
                    
            
        
    

    return

if __name__=='__main__':
    fdata()
