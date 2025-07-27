AI-Powered Document Scanner
This project uses a custom-trained YOLOv8 object detection model in combination with OpenCV to automatically find documents in images and correct their perspective.

It first locates a document within an image and then applies a series of image processing techniques to find its exact corners. 

Tech Stack
Python

OpenCV

YOLOv8 (Ultralytics/PyTorch)

NumPy

Python 3.8+

Install the required packages:

Bash

pip install ultralytics opencv-python numpy


Dataset URL: https://universe.roboflow.com/lung-x8el1/document-detection-v2
