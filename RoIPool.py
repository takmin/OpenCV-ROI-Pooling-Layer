import cv2
import torch
import torchvision
import torchvision.transforms as transforms

# This network has only ROI Pooling layer
model = torchvision.ops.RoIPool([4,2],1.0)

# Load Image
img = cv2.imread('2007_000720.jpg')

# Convert an image to tensor
input = transforms.ToTensor()(img)

# Add batch dimention
input = input.unsqueeze(0)

# ROI
rois = torch.tensor([[0, 216, 112, 304, 267]], dtype=torch.float)

# Test Model with image and roi
output = model(input, rois)

# Print test outputs
print("output: ")
print(output)

# Save Model as ONNX
filename = 'roi_pool.onnx'
torch.onnx.export(model, (input,rois), filename,input_names=['input','boxes'], output_names=['output'], opset_version=11)
