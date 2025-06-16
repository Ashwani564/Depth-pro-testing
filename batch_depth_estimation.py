from PIL import Image
import depth_pro
import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
import torch # GPU: Import torch

# --- Configuration ---
INPUT_FOLDER = "input_images" # CHANGE THE FOLDER NAME
OUTPUT_FOLDER = "output_results" # CHANGE THE FOLDER
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
LEGEND_WIDTH = 300
BATCH_SIZE = 4 # Process 4 images at a time # CHANGE THE BATCH SIZE 
# --- End Configuration ---

def process_batch(batch_paths, yolo_model, depth_model, transform, output_folder, device):
    """
    Processes a batch of images for object detection and depth estimation.
    YOLO detection is run on the entire batch at once for efficiency.
    """
    # This set is defined here to be used for each image in the batch
    CLASSES_TO_IGNORE = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Safety Vest", "Hardhat"}

    # 1. Load all images in the batch
    batch_images = []
    valid_paths = []
    for path in batch_paths:
        img = cv2.imread(path)
        if img is not None:
            batch_images.append(img)
            valid_paths.append(path)
        else:
            print(f"Error: Could not read image {path}, skipping.")

    if not batch_images:
        print("No valid images in this batch.")
        return

    # 2. Run batched YOLO object detection for performance
    print(f"Running YOLO detection on a batch of {len(batch_images)} images...")
    yolo_results_list = yolo_model(batch_images)
    print("YOLO detection for batch complete.")

    # 3. Process and save each image in the batch individually
    for i, yolo_results in enumerate(yolo_results_list):
        image_path = valid_paths[i]
        yolo_input_img = batch_images[i]

        print(f"--- Processing results for: {os.path.basename(image_path)} ---")
        base_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(base_filename)

        output_detection_path = os.path.join(output_folder, f"{name}_detection_depth{ext}")
        output_depth_map_path = os.path.join(output_folder, f"{name}_depth_map{ext}")

        try:
            img_h, img_w = yolo_input_img.shape[:2]
            final_image = np.ones((img_h, img_w + LEGEND_WIDTH, 3), dtype=np.uint8) * 255
            final_image[0:img_h, 0:img_w] = yolo_input_img

            # Extract detected objects from the pre-computed YOLO results
            detected_objects = []
            boxes = yolo_results.boxes.xyxy.cpu().numpy()
            classes = yolo_results.boxes.cls.cpu().numpy()

            for box, cls_idx in zip(boxes, classes):
                class_name = yolo_results.names[int(cls_idx)]
                if class_name in CLASSES_TO_IGNORE:
                    continue  # Skip ignored classes
                x1, y1, x2, y2 = map(int, box[:4])
                detected_objects.append({'box': (x1, y1, x2, y2), 'name': class_name})

            # Depth Estimation (still processed one-by-one)
            image_for_depth, _, f_px = depth_pro.load_rgb(image_path)
            depth_input_transformed = transform(image_for_depth).to(device)
            prediction = depth_model.infer(depth_input_transformed, f_px=f_px)
            depth = prediction["depth"]
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
    # CHANGE THE PATH
    yolo_model_instance = YOLO(r'C:\Users\%USERPROFILE%\Documents\yolo-med\runs\detect\train\weights\best.pt')
    yolo_model_instance.to(device)
    print("YOLO model loaded.")

    print("Loading Depth Pro model and transforms...")
    depth_model_instance, transform_instance = depth_pro.create_model_and_transforms()
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

    # --- MODIFICATION: Loop through the images in batches ---
    for i in range(0, len(image_files), BATCH_SIZE):
        batch_paths = image_files[i:i + BATCH_SIZE]
        print(f"\nProcessing batch {i // BATCH_SIZE + 1}...")
        process_batch(batch_paths, yolo_model_instance, depth_model_instance, transform_instance, OUTPUT_FOLDER, device)

    print("\nAll processing complete.")

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
