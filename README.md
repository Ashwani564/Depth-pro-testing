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
cd ml-depth-pro
Use code with caution.
Markdown
(Note: The scripts in this current project (like depth_test.py and batch_depth_estimation.py) should ideally be placed within or alongside this cloned ml-depth-pro directory, or ensure Python can find the depth_pro module from its location).
2. Download Pre-trained Depth Model Checkpoint
The ml-depth-pro repository provides a script (get_pretrained_models.sh) to download the model checkpoint. This script uses wget.
For Windows Users (or systems without wget/source readily available):
If you are on Windows, you can use PowerShell to download the pre-trained model directly.
First, ensure you have a checkpoints directory in your ml-depth-pro (or project) root. If not, create it:
mkdir checkpoints
Use code with caution.
Powershell
Then, run the following command in PowerShell to download the depth_pro.pt file into the checkpoints directory:
Invoke-WebRequest -Uri "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt" -OutFile "checkpoints\depth_pro.pt"
Use code with caution.
Powershell
For Linux/macOS Users:
You can use the provided shell script from within the cloned ml-depth-pro directory:
source get_pretrained_models.sh
Use code with caution.
Bash
3. Install Required Python Packages
This project requires ultralytics for YOLO object detection, opencv-python for image processing, Pillow and numpy. The ml-depth-pro model itself has its own dependencies which are usually covered by standard scientific Python packages (like PyTorch, torchvision - ensure these are installed, preferably in a virtual environment).
Install ultralytics (which often brings in PyTorch and other essentials if not present):
pip install ultralytics opencv-python Pillow numpy
Use code with caution.
Bash
If you encounter issues with PyTorch, please refer to the official PyTorch website for installation instructions specific to your system and CUDA version (if applicable).
4. Prepare Your Images (for batch_depth_estimation.py)
Create a folder named input_images in your project directory.
Place all the images you want to process into this input_images folder.
5. Run the Scripts
a) Single Image Test (depth_test.py)
This script is typically for testing the setup with a single, hardcoded image path.
You might need to modify the image_path variable inside depth_test.py to point to an image you want to test.
Ensure the YOLO model (e.g., yolo11s.pt or yolov8s.pt) is available in the same directory or the path is correctly specified in the script. The script will attempt to download it if it's a standard Ultralytics model name.
Run the script:
python depth_test.py
Use code with caution.
Bash
This will display the image with detected persons and their estimated depths, and also save an output image and a depth map.
b) Batch Processing (batch_depth_estimation.py)
This script processes all images from the input_images folder and saves the results (detection with depth overlay, and a separate depth colormap image) to an output_results folder.
Ensure the INPUT_FOLDER and OUTPUT_FOLDER variables at the top of batch_depth_estimation.py are set correctly (default is input_images and output_results).
Configure Object Detection:
The script uses a dictionary MODEL_CLASSES_TO_USER_LABELS to map classes detected by your YOLO model to user-friendly labels.
When you run batch_depth_estimation.py for the first time, it will print the available class names from your loaded YOLO model.
You MUST inspect these printed class names and update the KEYS of the MODEL_CLASSES_TO_USER_LABELS dictionary accordingly to detect and label the objects you are interested in (e.g., "person", "truck", "car", etc.).
Run the script:
python batch_depth_estimation.py
Use code with caution.
Bash
Check the output_results folder for the processed images.
How it Works
Object Detection: YOLO model identifies specified objects in the input image and provides their bounding boxes.
Depth Estimation: The ML-Depth-Pro model processes the same image (or its RGB representation) to generate a dense depth map.
Combining Results:
For each detected object, the script finds the center of its bounding box.
The depth value at this center coordinate is retrieved from the depth map generated by ML-Depth-Pro.
This depth value, along with the object's label, is then overlaid on the original image.
Notes
The accuracy of the depth estimation depends on the ML-Depth-Pro model and the quality of the input image.
The accuracy of object detection depends on the chosen YOLO model and its training data.
Ensure paths to models and images are correctly set within the Python scripts if you deviate from the default folder structure.
Consider using a Python virtual environment to manage dependencies.
