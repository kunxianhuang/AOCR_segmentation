#!/usr/bin/env python3.9

import os,sys
from glob import glob
import click
import shutil
from tqdm import tqdm
import nibabel as nib

classp_threshold = 0.5 # threshold of class prediction 

@click.command()
@click.option('-l','--label_dir',default='./runs/segment/predict3/labels/' ,help='label directory')
@click.option('-i','--image_dir',default='./data/3_Test1_Image/',help='nii test directory')
@click.option('-i','--save_file',default='./data/test_submission.csv',help='test submission file name')
def test_submit(label_dir,image_dir,save_file):
    '''read false-positive labels and save its image and label txt to train/val data'''
    nii_filelist = glob(image_dir+'*.nii.gz')
    #print(mask_filelist)                                                                                                

    # ZxFF51EAF10B09D34CAEEEA49B2A8310C1011E389C1AFE6C8B_97.txt

    label_array = []
    with open(save_file,'w+') as savef:
        savef.write("id,label\n")
        for ix, nii_file in enumerate(tqdm(nii_filelist)):
            filename = os.path.basename(nii_file)
            nii_name = filename.replace(".nii.gz",'')
            n1_img = nib.load(nii_file)
            nii_slices = n1_img.get_fdata()
        
            if isinstance(n1_img, type(None)):
                print('File {} does not include nii image'.format(nii_file))
                continue
        
            slice_num = nii_slices.shape[2]
            write_txt = ""

            nii_flag = 0
            for slice_i in range(slice_num):
                slice_flag = 0
                label_fname = "{}{}_{}.txt".format(label_dir,nii_name,slice_i)
                #print(label_fname)
                if os.path.exists(label_fname):
                    with open(label_fname,'r') as flabel:
                        for fp_seg in  flabel.readlines():
                            fp_seg_list = fp_seg.split()
                            cls = int(fp_seg_list[0])
                            if cls>0:
                                #print("Detected!!")
                                # probability of class 1
                                classp = float(fp_seg_list[-1])
                            
                                # save image and label txt
                                if classp > classp_threshold:
                                    slice_flag = 1
                                    nii_flag = 1
                slice_txt = "{}_{},{}\n".format(nii_name,slice_i,slice_flag)
                write_txt = write_txt + slice_txt # append id, label txt for each slice

            
            # nii file tag
            nii_label_txt = "{},{}\n".format(nii_name,nii_flag)
            write_txt = nii_label_txt+write_txt
            savef.write(write_txt)
        
    

    return

if __name__=='__main__':
    test_submit()
