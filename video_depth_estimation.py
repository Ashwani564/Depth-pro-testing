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
from tqdm import tqdm
import threading
from queue import Queue, Empty

# --- MLX Configuration (No changes) ---
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.machine() == 'arm64'
MLX_AVAILABLE = False
if IS_APPLE_SILICON:
    try:
        import mlx.core as mx
        MLX_AVAILABLE = True
        print("✅ MLX is available.")
    except ImportError:
        print("⚠️ MLX not found. To enable the MLX demonstration on your Mac, run: pip install mlx")

# --- Configuration (No changes) ---
LEGEND_WIDTH = 300

YOLO_MODEL_PATH = '/Users/ashwani/Desktop/YOLOV11M-Construction/runs/detect/train/weights/best.pt'
CLASSES_TO_IGNORE = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Safety Vest", "Hardhat"}
OUTPUT_FOLDER = "output_video"


# --- OPTIMIZATION 1: PROCESSING IN A SEPARATE THREAD ---
class FrameProcessor(threading.Thread):
    def __init__(self, frame_queue, result_queue, args, yolo_model, depth_model, transform, device):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.args = args
        self.yolo_model = yolo_model
        self.depth_model = depth_model
        self.transform = transform
        self.device = device
        self.running = True
        self.use_half = self.device != 'cpu' # Enable FP16 only on GPU/MPS

    def run(self):
        # OPTIMIZATION 2: USE HALF-PRECISION (FP16) on compatible devices
        if self.use_half:
            self.yolo_model.half()
            self.depth_model.half()

        while self.running:
            try:
                # Get a frame from the input queue
                frame_data = self.frame_queue.get(timeout=1)
                if frame_data is None:  # Sentinel value to stop
                    self.running = False
                    continue

                original_frame, frame_count = frame_data

                # --- Frame Processing Logic (moved from main loop) ---
                if args.width > 0:
                    aspect_ratio = original_frame.shape[0] / original_frame.shape[1]
                    new_h = int(args.width * aspect_ratio)
                    processing_frame = cv2.resize(original_frame, (args.width, new_h), interpolation=cv2.INTER_AREA)
                else:
                    processing_frame = original_frame

                img_h, img_w = processing_frame.shape[:2]
                
                # Convert frame and create tensors
                rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
                depth_input_transformed = self.transform(Image.fromarray(rgb_frame)).unsqueeze(0).to(self.device)
                f_px_tensor = torch.tensor([float(img_w)], device=self.device)
                
                if self.use_half:
                    depth_input_transformed = depth_input_transformed.half()

                # Inference
                with torch.no_grad():
                    yolo_results = self.yolo_model(rgb_frame, device=self.device, verbose=False, conf=0.4, half=self.use_half)[0]
                    prediction = self.depth_model.infer(depth_input_transformed, f_px=f_px_tensor)

                depth_np = prediction["depth"].squeeze().cpu().numpy()
                depth_np_resized = cv2.resize(depth_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

                # Visualization
                final_image = np.ones((img_h, img_w + LEGEND_WIDTH, 3), dtype=np.uint8) * 255
                final_image[0:img_h, 0:img_w] = processing_frame

                boxes = yolo_results.boxes.xyxy.cpu().numpy()
                classes = yolo_results.boxes.cls.cpu().numpy()
                detection_id = 1
                for box, cls_idx in zip(boxes, classes):
                    class_name = self.yolo_model.names[int(cls_idx)]
                    if class_name in CLASSES_TO_IGNORE: continue
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    center_x = np.clip((x1 + x2) // 2, 0, img_w - 1)
                    center_y = np.clip((y1 + y2) // 2, 0, img_h - 1)
                    depth_value = depth_np_resized[center_y, center_x]
                    cv2.putText(final_image, str(detection_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    legend_text = f"{detection_id}: {class_name} - {depth_value:.2f}m"
                    cv2.putText(final_image, legend_text, (img_w + 10, 50 + (detection_id - 1) * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    detection_id += 1

                depth_min, depth_max = depth_np_resized.min(), depth_np_resized.max()
                if depth_max - depth_min > 0:
                    inv_depth_normalized = 1.0 - (depth_np_resized - depth_min) / (depth_max - depth_min)
                    depth_colormap = cv2.applyColorMap((inv_depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                else:
                    depth_colormap = np.zeros_like(processing_frame)

                # Put the results into the output queue
                self.result_queue.put((final_image, depth_colormap, frame_count))

            except Empty:
                # Queue was empty, just continue waiting
                continue

    def stop(self):
        self.running = False
        self.frame_queue.put(None) # Send sentinel to unblock queue.get()


def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run object detection and depth estimation on video.")
    parser.add_argument("-i", "--input", type=str, help="Path to the input video file. If not specified, webcam is used.")
    parser.add_argument("-r", "--record", action='store_true', help="Enable recording of webcam feed. Ignored if --input is used.")
    parser.add_argument("--width", type=int, default=854, help="Resize frame to this width for processing. Default: 854.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process 1 frame and skip N-1 frames. Set to 1 for best responsiveness. Default: 1.")
    global args
    args = parser.parse_args()

    # --- Device Selection ---
    if torch.cuda.is_available():
        device = 'cuda'
    elif IS_APPLE_SILICON and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device.upper()}")
    if device != 'cpu':
        print("✅ Half-precision (FP16) will be used for acceleration.")

    # --- Model Loading ---
    print("Loading models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    depth_model, transform = depth_pro.create_model_and_transforms()
    depth_model.to(device).eval()
    print("Models loaded.")

    # --- 2. Input and Output Setup ---
    is_video_file = args.input is not None
    is_recording = args.record or is_video_file
    input_source = args.input if is_video_file else 0

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{input_source}'.")
        return

    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Video Writer Setup ---
    writer = None
    depth_writer = None
    if is_recording:
        # (Setup logic remains the same as your original code)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) if is_video_file else 20
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        if is_video_file:
            base_name = os.path.basename(args.input)
            name, _ = os.path.splitext(base_name)
            output_path = os.path.join(OUTPUT_FOLDER, f"{name}_processed.mp4")
            depth_output_path = os.path.join(OUTPUT_FOLDER, f"{name}_depth_map.mp4")
        else:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(OUTPUT_FOLDER, f"webcam_{timestr}_processed.mp4")
            depth_output_path = os.path.join(OUTPUT_FOLDER, f"webcam_{timestr}_depth_map.mp4")

        # Writers are created with the *resized* processing dimensions for better performance
        # We will resize back up ONLY if necessary at the very end.
        proc_w = args.width if args.width > 0 else original_w
        proc_h = int(proc_w * (original_h/original_w))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (proc_w + LEGEND_WIDTH, proc_h))
        depth_writer = cv2.VideoWriter(depth_output_path, fourcc, fps, (proc_w, proc_h))
        print(f"Saving processed video to: {output_path}")


    # --- 3. Threading and Queue Setup ---
    # Queues should have a max size to prevent memory overflow if the processor can't keep up
    frame_queue = Queue(maxsize=4) 
    result_queue = Queue(maxsize=4)

    processor_thread = FrameProcessor(frame_queue, result_queue, args, yolo_model, depth_model, transform, device)
    processor_thread.start()

    # --- 4. Main Loop (I/O and Display) ---
    pbar = None
    if is_video_file:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processing Video")

    frame_count = 0
    fps_timer = time.time()
    display_fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            # --- Frame Reading and Queueing ---
            frame_count += 1
            if frame_count % args.frame_skip == 0 and not frame_queue.full():
                # Put the original frame and its number into the queue for the processor
                frame_queue.put((frame, frame_count))

            # --- Result Displaying ---
            try:
                # Get the latest processed result without blocking
                final_image, depth_colormap, processed_frame_num = result_queue.get_nowait()
                
                if writer:
                    # NOTE: Writing the smaller, processed frame is much faster.
                    # If you absolutely need original resolution, resize here:
                    # final_image = cv2.resize(final_image, (original_w + LEGEND_WIDTH, original_h))
                    # depth_colormap = cv2.resize(depth_colormap, (original_w, original_h))
                    writer.write(final_image)
                    depth_writer.write(depth_colormap)

                if not is_video_file:
                    # Add FPS counter to display
                    if time.time() - fps_timer > 1:
                        display_fps = result_queue.qsize() / (time.time() - fps_timer)
                        fps_timer = time.time()
                    
                    cv2.putText(final_image, f"FPS: {display_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Real-time Depth Detection', final_image)
                    cv2.imshow('Depth Map', depth_colormap)

                if pbar: pbar.update(processed_frame_num - pbar.n)

            except Empty:
                # No new result is ready yet, just continue
                pass

            if not is_video_file and cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # --- 5. Cleanup ---
        print("\nClosing application...")
        processor_thread.stop()
        processor_thread.join() # Wait for the thread to finish
        if pbar: pbar.close()
        cap.release()
        if writer: writer.release()
        if depth_writer: depth_writer.release()
        cv2.destroyAllWindows()
        print("Resources released.")


if __name__ == '__main__':
    main()
