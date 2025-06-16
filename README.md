# Depth-pro-testing

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
### 2. Download Pre-trained Depth Model Checkpoint
The ml-depth-pro repository provides a script (get_pretrained_models.sh) to download the model checkpoint. This script uses wget.
For Windows Users (or systems without wget/source readily available):
If you are on Windows, you can use PowerShell to download the pre-trained model directly.
First, ensure you have a checkpoints directory in your ml-depth-pro (or project) root. If not, create it:
```bash
mkdir checkpoints
```
#### Powershell
Then, run the following command in mediocre PowerShell to download the depth_pro.pt file into the checkpoints directory:
```bash
Invoke-WebRequest -Uri "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt" -OutFile "checkpoints\depth_pro.pt"
```

####For Linux/macOS Users:
You can use the provided shell script from within the cloned ml-depth-pro directory:
```bash
source get_pretrained_models.sh
```
### 3. Install Required Python Packages
This project requires ultralytics for YOLO object detection, opencv-python for image processing, Pillow and numpy. 
Install ultralytics (which often brings in PyTorch and other essentials if not present):
```bash
pip install ultralytics opencv-python Pillow numpy
```

##4. Prepare Your Images (for batch_depth_estimation.py)
Create a folder named input_images in your project directory.
Place all the images you want to process into this input_images folder.
###5. Run the Scripts
###a) Single Image Test (depth_test.py)
This script is typically for testing the setup with a single, hardcoded image path.
```bash
python depth_test.py
```
You might need to modify the image_path variable inside depth_test.py to point to an image you want to test.

Ensure the YOLO model (e.g., yolo11s.pt or yolov8s.pt) is available in the same directory or the path is correctly specified in the script. The script will attempt to download it if it's a standard Ultralytics model name.
Run the script:

This will display the image with detected persons and their estimated depths, and also save an output image and a depth map.
###b) Batch Processing (batch_depth_estimation.py)
This script processes all images from the input_images folder and saves the results (detection with depth overlay, and a separate depth colormap image) to an output_results folder.
```bash
python batch_depth_estimation.py
```
