# Robot-Bottle-Picking
Computer vision scripts for liquor bottle detection for robotic warehouse automation

## Description:
This repository is part of a bigger project to develop an industrial robot that can detect, classify and pick liquor bottles off warehouse shelves. For ease of viewing, I have converted most Python scripts to Jupyter Notebooks. Keep in mind that these have mostly been optimized for command-line use.

Contents are as follows, and can be viewed in this order:
<br>
__bottle-detection/__
1. _baseline-model_ssd-mobilenet.ipynb_
  - Using a pre-trained SSD-MobileNet model (trained on COCO dataset) to detect bottles in our images.
2. Training a model on our own images (many thanks to Dat Tran for his [Racoon Detector!](https://github.com/datitran/raccoon_dataset))
  - _train_test_img_split.py_
  - _xml_to_csv.py_
  - _generate_tfrecord.py_
  - _export_inference_graph.py_
  - _eval.py_
  - _images/_ : contains our train and test images, as well as our results
  - _models/_ : contains the models we create
  - _training/_ : contains training checkpoints
  - _eval/_ : contains evaluation results from training our model
  - _data/_ : contains generated tfrecords and csv
3. _bottle-detection_distance-estimation_.ipynb
  - Using our newly-trained model (trained on COCO + our own images) to detect bottles, and then estimate the distance from the camera to each bounding box.
<br>
__stereo/__
1. Creating disparity maps
- _stereo_match2.py_
- _stereo_match3.py_
- _stereo_testing.ipynb_

## Dependencies:
- Tensorflow
- Tensorflow Object Detection API
- OpenCV 3

## Installation:
For installation instructions, please follow:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
For installation instructions on Windows, please follow:
https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b
