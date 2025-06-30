import argparse
from PIL import Image
import depth_pro
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import platform
import time
import os
from tqdm import tqdm # For progress bar

# --- MLX Configuration for macOS ---
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.machine() == 'arm64'
MLX_AVAILABLE = False
if IS_APPLE_SILICON:
    try:
        import mlx.core as mx
        MLX_AVAILABLE = True
        print("✅ MLX is available.")
    except ImportError:
        print("⚠️ MLX not found. To enable the MLX demonstration on your Mac, run: pip install mlx")
# --- End MLX Configuration ---


# --- Configuration ---
LEGEND_WIDTH = 300
YOLO_MODEL_PATH = '/Users/ashwani/Desktop/YOLOV11M-Construction/runs/detect/train/weights/best.pt'
CLASSES_TO_IGNORE = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Safety Vest", "Hardhat"}
OUTPUT_FOLDER = "output_video"
# --- End Configuration ---


def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Real-time Object Detection and Metric Depth Estimation.")
    parser.add_argument("-i", "--input", type=str, help="Path to the input video file. If not specified, webcam will be used.")
    parser.add_argument("-r", "--record", action='store_true', help="Enable recording of webcam feed. Ignored if --input is used.")
    args = parser.parse_args()

    # --- Device Selection ---
    if torch.cuda.is_available():
        device = 'cuda'
    elif IS_APPLE_SILICON and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device.upper()}")

    # --- Model Loading ---
    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("YOLO model loaded.")

    print("Loading Depth Pro model and transforms...")
    depth_model, transform = depth_pro.create_model_and_transforms()
    depth_model.to(device)
    depth_model.eval()
    print("Depth Pro model and transforms loaded.")

    # --- 2. Input and Output Setup ---
    is_video_file = args.input is not None
    is_recording = args.record or is_video_file

    if is_video_file:
        input_source = args.input
        print(f"Processing video file: {input_source}")
    else:
        input_source = 0
        print("Opening webcam...")
        if args.record:
            print("Webcam recording is ENABLED.")

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{input_source}'.")
        return

    # Video Writer Initialization (if recording or processing a file)
    writer = None
    depth_writer = None
    if is_recording:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use source FPS for video files, or a default for webcams
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 20 # A reasonable default for webcams
            print(f"Warning: Could not determine webcam FPS. Defaulting to {fps} FPS for recording.")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
        
        if is_video_file:
            base_name = os.path.basename(args.input)
            name, _ = os.path.splitext(base_name)
            output_path = os.path.join(OUTPUT_FOLDER, f"{name}_processed.mp4")
            depth_output_path = os.path.join(OUTPUT_FOLDER, f"{name}_depth_map.mp4")
        else: # Recording from webcam
            timestr = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(OUTPUT_FOLDER, f"webcam_{timestr}_processed.mp4")
            depth_output_path = os.path.join(OUTPUT_FOLDER, f"webcam_{timestr}_depth_map.mp4")

        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w + LEGEND_WIDTH, frame_h))
        depth_writer = cv2.VideoWriter(depth_output_path, fourcc, fps, (frame_w, frame_h))
        print(f"Saving processed video to: {output_path}")
        print(f"Saving depth map video to: {depth_output_path}")

    # --- 3. Processing Loop ---
    prev_time = 0
    pbar = None
    if is_video_file:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processing Video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        img_h, img_w = frame.shape[:2]

        # --- Model Inference (same as before) ---
        yolo_results = yolo_model(frame, device=device, verbose=False)[0]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        depth_input_transformed = transform(pil_image).unsqueeze(0).to(device)
        f_px_tensor = torch.tensor([float(img_w)], device=device)

        with torch.no_grad():
            prediction = depth_model.infer(depth_input_transformed, f_px=f_px_tensor)
        
        depth_np = prediction["depth"].squeeze().cpu().numpy()
        depth_np_resized = cv2.resize(depth_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        # --- Visualization (same as before) ---
        final_image = np.ones((img_h, img_w + LEGEND_WIDTH, 3), dtype=np.uint8) * 255
        final_image[0:img_h, 0:img_w] = frame

        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        classes = yolo_results.boxes.cls.cpu().numpy()
        
        detection_id = 1
        legend_y_start = 50
        line_height = 25
        for box, cls_idx in zip(boxes, classes):
            class_name = yolo_results.names[int(cls_idx)]
            if class_name in CLASSES_TO_IGNORE:
                continue
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            center_x = np.clip((x1 + x2) // 2, 0, img_w - 1)
            center_y = np.clip((y1 + y2) // 2, 0, img_h - 1)
            depth_value = depth_np_resized[center_y, center_x]
            label_text = str(detection_id)
            cv2.putText(final_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            legend_text = f"{detection_id}: {class_name} - {depth_value:.2f}m"
            cv2.putText(final_image, legend_text, (img_w + 10, legend_y_start + (detection_id - 1) * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            detection_id += 1

        # Depth Map Colormap
        depth_min, depth_max = depth_np_resized.min(), depth_np_resized.max()
        if depth_max - depth_min > 0:
            inv_depth_normalized = 1.0 - (depth_np_resized - depth_min) / (depth_max - depth_min)
            depth_colormap = cv2.applyColorMap((inv_depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        else:
            depth_colormap = np.zeros_like(frame)

        # --- 4. Output Handling ---
        # Write to file if recording is enabled
        if is_recording:
            writer.write(final_image)
            depth_writer.write(depth_colormap)
        
        # Display preview if not processing a file
        if not is_video_file:
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(final_image, f"FPS: {int(fps)}", (img_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Real-time Depth Detection', final_image)
            cv2.imshow('Depth Map', depth_colormap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if pbar:
            pbar.update(1)

    # --- 5. Cleanup ---
    print("\nClosing application...")
    if pbar:
        pbar.close()
    cap.release()
    if writer:
        writer.release()
    if depth_writer:
        depth_writer.release()
    cv2.destroyAllWindows()
    print("Resources released.")


if __name__ == '__main__':
    # You might need to install tqdm: pip install tqdm
    main()