from PIL import Image
import depth_pro
import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob

# --- Configuration ---
INPUT_FOLDER = "input_images"  # Folder containing your images
OUTPUT_FOLDER = "output_results" # Folder where results will be saved
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff') # Common image extensions
# --- End Configuration ---

def process_image(image_path, yolo_model, depth_model, transform, output_folder):
    """
    Processes a single image for person detection and depth estimation,
    saving the results.
    """
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

        # --- YOLO Person Detection ---
        results = yolo_model(yolo_input_img)
        person_boxes = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, cls_idx in zip(boxes, classes):
                if result.names[int(cls_idx)] == 'person':
                    x1, y1, x2, y2 = map(int, box[:4])
                    person_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(yolo_input_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- Depth Estimation ---
        image_for_depth, _, f_px = depth_pro.load_rgb(image_path) # Use original image path for depth_pro
        depth_input_transformed = transform(image_for_depth)

        prediction = depth_model.infer(depth_input_transformed, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m]
        depth_np = depth.squeeze().cpu().numpy()

        # Resize depth_np to match yolo_input_img dimensions
        if depth_np.shape[0] != yolo_input_img.shape[0] or depth_np.shape[1] != yolo_input_img.shape[1]:
            depth_np_resized = cv2.resize(depth_np, (yolo_input_img.shape[1], yolo_input_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            depth_np_resized = depth_np

        # --- Overlay Depth Information on Detections ---
        for x1, y1, x2, y2 in person_boxes:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            center_y = np.clip(center_y, 0, depth_np_resized.shape[0] - 1)
            center_x = np.clip(center_x, 0, depth_np_resized.shape[1] - 1)

            depth_value = depth_np_resized[center_y, center_x]

            text = f'Depth: {depth_value:.2f}m'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            text_draw_x = x1
            text_draw_y = y1 - 10

            if text_draw_y < text_size[1] + 5:
                text_draw_y = y1 + text_size[1] + 10

            rect_x1_bg = text_draw_x
            rect_y1_bg = text_draw_y - text_size[1] - 5
            rect_x2_bg = text_draw_x + text_size[0]
            rect_y2_bg = text_draw_y + 5

            cv2.rectangle(yolo_input_img, (rect_x1_bg, rect_y1_bg), (rect_x2_bg, rect_y2_bg), (0, 0, 0), -1)
            cv2.putText(yolo_input_img, text, (text_draw_x, text_draw_y), font, font_scale, (255, 255, 255), font_thickness)

        cv2.imwrite(output_detection_path, yolo_input_img)
        print(f"Saved detection image to: {output_detection_path}")

        # --- Generate and Save Inverted Depth Map ---
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
    # --- Load Models (once) ---
    print("Loading YOLO model...")
    yolo_model_instance = YOLO('yolo11s.pt') # Using a more specific name
    print("YOLO model loaded.")

    print("Loading Depth Pro model and transforms...")
    depth_model_instance, transform_instance = depth_pro.create_model_and_transforms()
    depth_model_instance.eval()
    print("Depth Pro model and transforms loaded.")
    # --- End Model Loading ---

    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    # Get list of images to process
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    if not image_files:
        print(f"No images found in {INPUT_FOLDER} with extensions {IMAGE_EXTENSIONS}")
        return

    print(f"Found {len(image_files)} images to process.")

    for image_file_path in image_files:
        process_image(image_file_path, yolo_model_instance, depth_model_instance, transform_instance, OUTPUT_FOLDER)

    print("Processing complete.")

if __name__ == '__main__':
    # Create a dummy input folder and an example image for testing if they don't exist
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created dummy input folder: {INPUT_FOLDER}")
        # Create a simple dummy image if opencv can write it
        try:
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(INPUT_FOLDER, "example.jpg"), dummy_image)
            print(f"Created dummy image: {os.path.join(INPUT_FOLDER, 'example.jpg')}")
            print(f"Please replace 'example.jpg' or add your images to the '{INPUT_FOLDER}' directory.")
        except Exception as e:
            print(f"Could not create dummy image: {e}. Please create the input folder and add images manually.")


    main()