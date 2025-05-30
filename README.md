# Depth-pro-testing
# Depth Estimation with Object Detection (YOLO + ML-Depth-Pro)

This project demonstrates how to combine the ML-Depth-Pro model from Apple for depth estimation with YOLO (You Only Look Once) for object detection. It allows you to identify objects (e.g., people, and other configurable classes) in an image and estimate their depth from the camera.

## Original Model

This project utilizes Apple's ML-Depth-Pro model. For more details on the original model, its capabilities, and research, please visit their official repository:
[https://github.com/apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)

## Setup Instructions

### 1. Clone the ML-Depth-Pro Repository

First, you need to clone the official `ml-depth-pro` repository from Apple. This will provide the necessary source code and scripts for the depth model.

```bash
git clone https://github.com/apple/ml-depth-pro.git
```
```bash
cd ml-depth-pro
```
####Use code with caution.
###2. Download Pre-trained Depth Model Checkpoint
The ml-depth-pro repository provides a script (get_pretrained_models.sh) to download the model checkpoint. This script uses wget.
For Windows Users (or systems without wget/source readily available):
If you are on Windows, you can use PowerShell to download the pre-trained model directly.
First, ensure you have a checkpoints directory in your ml-depth-pro (or project) root. If not, create it:
```bash
mkdir checkpoints
```
Use code with caution.
Powershell
Then, run the following command in PowerShell to download the depth_pro.pt file into the checkpoints directory:
```bash
Invoke-WebRequest -Uri "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt" -OutFile "checkpoints\depth_pro.pt"
```
Use code with caution.
Powershell
For Linux/macOS Users:
You can use the provided shell script from within the cloned ml-depth-pro directory:
```bash
source get_pretrained_models.sh
```
###3. Install Required Python Packages
This project requires ultralytics for YOLO object detection, opencv-python for image processing, Pillow and numpy. The ml-depth-pro model itself has its own dependencies which are usually covered by standard scientific Python packages (like PyTorch, torchvision - ensure these are installed, preferably in a virtual environment).
Install ultralytics (which often brings in PyTorch and other essentials if not present):
```bash
pip install ultralytics opencv-python Pillow numpy
```
If you encounter issues with PyTorch, please refer to the official PyTorch website for installation instructions specific to your system and CUDA version (if applicable).
###4. Prepare Your Images (for batch_depth_estimation.py)
Create a folder named input_images in your project directory.
Place all the images you want to process into this input_images folder.
###5. Run the Scripts
a) Single Image Test (depth_test.py)
This script is typically for testing the setup with a single, hardcoded image path.
You might need to modify the image_path variable inside depth_test.py to point to an image you want to test.
Ensure the YOLO model (e.g., yolo11s.pt or yolov8s.pt) is available in the same directory or the path is correctly specified in the script. The script will attempt to download it if it's a standard Ultralytics model name.
Run the script:
```bash
python depth_test.py
```
This will display the image with detected persons and their estimated depths, and also save an output image and a depth map.
b) Batch Processing (batch_depth_estimation.py)
This script processes all images from the input_images folder and saves the results (detection with depth overlay, and a separate depth colormap image) to an output_results folder.
Ensure the INPUT_FOLDER and OUTPUT_FOLDER variables at the top of batch_depth_estimation.py are set correctly (default is input_images and output_results).
Configure Object Detection:
