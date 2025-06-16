from PIL import Image
import depth_pro
import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
import torch # GPU: Import torch

# --- Configuration ---
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_results"
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
LEGEND_WIDTH = 300
# --- End Configuration ---

def process_image(image_path, yolo_model, depth_model, transform, output_folder, device): # GPU: Pass device as an argument
    """
    Processes a single image for object detection and depth estimation,
    saving the results with a numbered legend on the side.
    """
    # --- MODIFICATION: Define the set of classes to ignore ---
    CLASSES_TO_IGNORE = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Safety Vest", "Hardhat"}

    print(f"Processing: {image_path}")
    base_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(base_filename)

    output_detection_path = os.path.join(output_folder, f"{name}_detection_depth{ext}")
    output_depth_map_path = os.path.join(output_folder, f"{name}_depth_map{ext}")

    try:
        yolo_input_img = cv2.imread(image_path)
        if yolo_input_img is None:
            print(f"Error: Could not read image {image_path}")
            return

        img_h, img_w = yolo_input_img.shape[:2]

        final_image = np.ones((img_h, img_w + LEGEND_WIDTH, 3), dtype=np.uint8) * 255
        final_image[0:img_h, 0:img_w] = yolo_input_img

        # --- YOLO Object Detection ---
        # The model was already moved to the device, so this will run on the GPU
        results = yolo_model(yolo_input_img)
        detected_objects = []
        for result in results:
            # The .cpu() call is still necessary here to work with NumPy/OpenCV
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, cls_idx in zip(boxes, classes):
                class_name = result.names[int(cls_idx)]

                # --- MODIFICATION: Check if the detected class should be ignored ---
                if class_name in CLASSES_TO_IGNORE:
                    continue # Skip to the next detected object

                x1, y1, x2, y2 = map(int, box[:4])
                detected_objects.append({'box': (x1, y1, x2, y2), 'name': class_name})

        # --- Depth Estimation ---
        image_for_depth, _, f_px = depth_pro.load_rgb(image_path)
        # GPU: Move the input tensor to the same device as the model
        depth_input_transformed = transform(image_for_depth).to(device)

        # This will now run on the GPU
        prediction = depth_model.infer(depth_input_transformed, f_px=f_px)
        depth = prediction["depth"]
        # GPU: Move the result back to the CPU to use with NumPy and OpenCV
        depth_np = depth.squeeze().cpu().numpy()

        if depth_np.shape[0] != img_h or depth_np.shape[1] != img_w:
            depth_np_resized = cv2.resize(depth_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        else:
            depth_np_resized = depth_np

        detection_id = 1
        legend_y_start = 30
        line_height = 25

        for obj in detected_objects:
            x1, y1, x2, y2 = obj['box']
            class_name = obj['name']
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center_y = np.clip(center_y, 0, depth_np_resized.shape[0] - 1)
            center_x = np.clip(center_x, 0, depth_np_resized.shape[1] - 1)
            depth_value = depth_np_resized[center_y, center_x]

            label_text = str(detection_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            text_size, _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            text_draw_x = x1
            text_draw_y = y1 - 5
            if text_draw_y < text_size[1]:
                 text_draw_y = y1 + text_size[1] + 5

            cv2.putText(final_image, label_text, (text_draw_x, text_draw_y), font, font_scale, (0, 0, 255), font_thickness)

            legend_text = f"{detection_id}: {class_name} - {depth_value:.2f}m"
            legend_x_pos = img_w + 10
            legend_y_pos = legend_y_start + (detection_id - 1) * line_height
            cv2.putText(final_image, legend_text, (legend_x_pos, legend_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            detection_id += 1

        cv2.imwrite(output_detection_path, final_image)
        print(f"Saved detection image to: {output_detection_path}")

        depth_min_val = depth_np_resized.min()
        depth_max_val = depth_np_resized.max()
        if depth_max_val - depth_min_val > 0:
            depth_np_normalized = (depth_np_resized - depth_min_val) / (depth_max_val - depth_min_val)
        else:
            depth_np_normalized = np.zeros_like(depth_np_resized)

        inv_depth_np_normalized = 1.0 - depth_np_normalized
        depth_colormap = cv2.applyColorMap((inv_depth_np_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(output_depth_map_path, depth_colormap)
        print(f"Saved depth map to: {output_depth_map_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    # GPU: Set the device to 'cuda' if available, otherwise 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading YOLO model...")
    # The YOLO library automatically uses the available GPU if torch is set up correctly.
    yolo_model_instance = YOLO(r'C:\Users\am5082\Documents\yolo-med\runs\detect\train\weights\best.pt')
    yolo_model_instance.to(device) # GPU: Explicitly move model to device for good practice
    print("YOLO model loaded.")

    print("Loading Depth Pro model and transforms...")
    depth_model_instance, transform_instance = depth_pro.create_model_and_transforms()
    # GPU: Move the depth model to the selected device
    depth_model_instance.to(device)
    depth_model_instance.eval()
    print("Depth Pro model and transforms loaded.")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    if not image_files:
        print(f"No images found in {INPUT_FOLDER} with extensions {IMAGE_EXTENSIONS}")
        return

    print(f"Found {len(image_files)} images to process.")

    for image_file_path in image_files:
        # GPU: Pass the device to the processing function
        process_image(image_file_path, yolo_model_instance, depth_model_instance, transform_instance, OUTPUT_FOLDER, device)

    print("Processing complete.")

if __name__ == '__main__':
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created dummy input folder: {INPUT_FOLDER}")
        try:
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(INPUT_FOLDER, "example.jpg"), dummy_image)
            print(f"Created dummy image: {os.path.join(INPUT_FOLDER, 'example.jpg')}")
            print(f"Please replace 'example.jpg' or add your images to the '{INPUT_FOLDER}' directory.")
        except Exception as e:
            print(f"Could not create dummy image: {e}. Please create the input folder and add images manually.")

    main()
