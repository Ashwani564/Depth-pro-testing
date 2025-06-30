# Depth-pro-testing
This project integrates a custom-trained YOLOv11 model with Apple's ML-Depth-Pro for depth estimation in images, focusing on construction site scenarios.
## Update As Of June 30, 2025 (Incomplete info; will fill later):
```
python video_depth_estimation.py --input input_video/video.mp4 --width 640 --frame-skip 10
```

## Update As of June 16, 2025 (for past two weeks):
### Successful training and integration of Custom YOLO Model:
  ![MixCollage-16-Jun-2025-01-17-AM-5312](https://github.com/user-attachments/assets/6cfb825a-965e-4ff0-93e7-c3275de7bbba)
  ![MixCollage-16-Jun-2025-01-19-AM-9262](https://github.com/user-attachments/assets/5c6f1cd7-46ef-428c-90d1-2b270df1c111)
  ### Trained Three Different Models (are available to Download on Hugging Face along with their performance metric):
  1. YOLO 11S (MAP50 Accuracy: 88%): https://huggingface.co/Ashwani-0101/YOLO11S-Construction
  2. 2.YOLO 11M (MAP50 Accuracy: 90%): https://huggingface.co/Ashwani-0101/YOLOV11M-Construction
  3. YOLO 11L (MAP50 Accuracy: 90%): https://huggingface.co/Ashwani-0101/Yolo11L-construction
### I have integrated YOLO V11 M because of its significant inaccuracy in background along with faster inference than the large one.
### Changes in batch_depth_estimation:

  1. Improved the processing speed by implementing the batch size processing
  2. Integrated custom YOLO model for better detection
  3. Fixed cuda problem
  4. Implemented a method to label detection boxes as 1,2,3, making a side bar to mention the estimated depth.

### Warning: Change the path and batch size as per your needs in batch_dpeth_estimation. Do find "CHANGE THE" through script. It will get you where you nee to make changes 
  
## Update as of May 30, 2025
## Install Depth Pro Model

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

#### For Linux/macOS Users:
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

### 4. Prepare Your Images (for batch_depth_estimation.py)
Create a folder named input_images in your project directory.
Place all the images you want to process into this input_images folder.
###5. Run the Scripts
### a) Single Image Test (depth_test.py)
This script is typically for testing the setup with a single, hardcoded image path.
```bash
python depth_test.py
```
You might need to modify the image_path variable inside depth_test.py to point to an image you want to test.

Ensure the YOLO model (e.g., yolo11s.pt or yolov8s.pt) is available in the same directory or the path is correctly specified in the script. The script will attempt to download it if it's a standard Ultralytics model name.
Run the script:

This will display the image with detected persons and their estimated depths, and also save an output image and a depth map.
### b) Batch Processing (batch_depth_estimation.py)
This script processes all images from the input_images folder and saves the results (detection with depth overlay, and a separate depth colormap image) to an output_results folder.
```bash
python batch_depth_estimation.py
```
