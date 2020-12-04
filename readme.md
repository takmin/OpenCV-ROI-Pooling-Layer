# OpenCV DNN Module Custom Layer Sample (ROI Pooling)

These codes are sample files of ROI Pooling for torchvision and OpenCV

I confirmed these samples work on:
- OpenCV 4.5.0
- PyTorch 1.7.0
- torchvision 0.8.1

### RoIPool.py
PyTorch sample to create ROI Pooling layer, and to export it as onnx file.

### onnx2txt.py
Print onnx file parameters exported from RoIPool.py.
You need to install onnx module for python.

### RoIPool_CustomLayer.cpp
OpenCV code to use ROI Pooling layer.
OpenCV does not support ROI Pooling layer, so I implemented it as a custom layer.

2020/12/04 takmin