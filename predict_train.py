#!/usr/bin/env python3.9
from ultralytics import YOLO
from glob import glob
import os,sys
import click
from tqdm import tqdm


classp_threshold = 0.5



@click.command()
@click.option('-j','--train_jpgdir',default='./datasets/aocr-data/images/train/',help='training jpg directory')
@click.option('-l','--train_labeldir',default='./datasets/aocr-data/labels/train/',help='training label directory')
@click.option('-m','--loading_model',default='./runs/segment/train14/weights/best.pt',help='loading segmentation model')
def predict_train(train_jpgdir,train_labeldir,loading_model):

    # Load a model
    model = YOLO(loading_model)  # Load a first trained model

    train_jpgs = glob(train_jpgdir+"*.jpg")
    for ij, train_jpg in enumerate(tqdm(train_jpgs)):
        filename = os.path.basename(train_jpg)
        label_name = filename.replace(".jpg",'.txt')
        label_name = train_labeldir + label_name
        if os.path.exists(label_name):
            with open(label_name,'r') as lf:
                labels = lf.readlines()
                for label in labels:                
                    class_label = label.split()[0] # first int is class
                    class_label = int(class_label)
        else:
            print("Lable file {} does not exist.".format(label_name))
            continue
        
        results = model.predict(source=train_jpg, save=False, save_txt=False)  # return result of the jpg
        

        for i, result in enumerate(results):
            #masks = result.masks  # Masks object for segmentation masks outputs
            #print("mask {}\n".format(i))
            #print(masks)
            #print("class prob of {}\n".format(i))
            
            cls = result.boxes.cls.tolist()  # Predicted class 
            if len(cls)==0:
                continue
            cls = int(cls[0])
            prob = result.boxes.conf.tolist()[0] # Predicted probability
            #print(prob)
            if prob > classp_threshold:
                if cls==class_label: # compare prediction and label
                    continue
                else: #false positive or false negative
                    model.predict(source=train_jpg, save=True, save_txt=True)
            else:
                if class_label==1: # false negative
                    model.predict(source=train_jpg, save=True, save_txt=True)
                    
            
    
if __name__=='__main__':
    predict_train()
