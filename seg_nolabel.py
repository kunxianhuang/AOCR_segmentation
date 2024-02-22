#!/usr/bin/env python3.9
from ultralytics import YOLO
from glob import glob
import os,sys

# Load a model

model = YOLO("runs/segment/train12/weights/best.pt")  # Load a first trained model

nolabel_jpgdir= 'data/1_Train,Valid_Image/nolabel/'
list_nolabelf = glob(nolabel_jpgdir+'*.jpg')

for i, nolabelf in enumerate(list_nolabelf):
    results = model.predict(source=[nolabelf], save_txt=True, save_conf=True, conf=0.2)

