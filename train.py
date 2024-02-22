#!/usr/bin/env python3.9
from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n-seg.yaml")  # build a new model from scratch
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model from beginning
results = model.train(data='aocr.yaml', epochs=300, batch=128, imgsz=640)

# Validation
result = model.val()

# export model of first training
path = model.export(format="onnx")  # export the model to ONNX format

# Run batched inference on a list of images
results = model.predict(source=['datasets/aocr-data/images/val/Zx1060A72A4F5C5FFE63F395F263138C39C901CEA1B608B5D3_31.jpg',
                                'datasets/aocr-data/images/val/Zx09413D933AD838CE8DB00704AB349855FE4721FC26037F37_63.jpg'], save=True, save_txt=True)  # return a list of Results objects

for result in results:
    masks = result.masks  # Masks object for segmentation masks outputs
    print(masks)
