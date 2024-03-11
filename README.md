# AOCR_segmentation
For AOCR competition in kaggle (website:https://www.kaggle.com/competitions/aocr2024), this repository includes programs to data procession, pre-train, FP data, train, and test data submission.


# Data Pre-processing
* data/segananotation_label.py: As mask files are provided for appendicitis sample, we transfer it to contour samples which are used for instance segmentation. Slice image files are also saved.
* data/nolabel_image.py: Extract image slice from nii files of no appendicitis samples (normal appendix) and save them into directory. They will use for generating false positive samples.
* data/test_image.py: Extract slice image from test nii files and save them to test directory.
* fpdata.py: As we have pre-trained model for only appendicitis, we use it to predict false positive samples and save them into training and validation directories.

# Training
* begin_train.py: training with appendicitis samples (first training)
* aocr.yaml: yaml file for training/validation directory and classes names
* seg_nolabel.py: use the pre-trained model to predict false positive samples
* middle_train.py: training with appendicitis samples and false positive samples (normal appendix)

# Testing
* seg_test.py: Predict testing sample
* test_submit.py: Read predicted results and write a txt file to submit results.
* test_submitv2.py: Read predicted results and write a txt file to submit results and fix discontinuity of results by looking neighboring slices.
