from PIL import Image
import depth_pro
import cv2
import numpy as np
from ultralytics import YOLO

yolo_model = YOLO('yolo11s.pt')

image_path = "data/example.jpg"


yolo_input_img = cv2.imread(image_path) # Renamed to avoid confusion with depth_input

results = yolo_model(yolo_input_img)

person_boxes = []
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        if result.names[int(cls)] == 'person':
            x1, y1, x2, y2 = map(int, box[:4])
            person_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(yolo_input_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Load depth model and preprocessing transform
depth_model, transform = depth_pro.create_model_and_transforms()
depth_model.eval()

# It's better to load the image once for depth processing if it's the same as YOLO input
# Assuming image_path is the same for both YOLO and depth
image_for_depth, _, f_px = depth_pro.load_rgb(image_path)
depth_input_transformed = transform(image_for_depth)

prediction = depth_model.infer(depth_input_transformed, f_px=f_px)
depth = prediction["depth"] # Depth in [m]

depth_np = depth.squeeze().cpu().numpy()

# Ensure the depth map has the same dimensions as the yolo_input_img for coordinate mapping
# If not, resizing might be needed, but for now, we assume they are compatible or
# that the depth_pro.load_rgb and transform handle this.
# For simplicity, let's assume depth_np has dimensions that can be indexed by coordinates from yolo_input_img.
# If yolo_input_img was resized by YOLO internally, person_boxes coordinates might not directly map to original image/depth map.
# However, the provided code implies direct mapping.

# Resize depth_np to match yolo_input_img dimensions if they are different
# This is a common step if the depth model outputs a different resolution
if depth_np.shape[0] != yolo_input_img.shape[0] or depth_np.shape[1] != yolo_input_img.shape[1]:
    depth_np_resized = cv2.resize(depth_np, (yolo_input_img.shape[1], yolo_input_img.shape[0]), interpolation=cv2.INTER_NEAREST)
else:
    depth_np_resized = depth_np


for x1, y1, x2, y2 in person_boxes:
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Ensure center coordinates are within the bounds of the resized depth map
    center_y = np.clip(center_y, 0, depth_np_resized.shape[0] - 1)
    center_x = np.clip(center_x, 0, depth_np_resized.shape[1] - 1)

    depth_value = depth_np_resized[center_y, center_x]

    text = f'Depth: {depth_value:.2f}m'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 # Adjusted for potentially smaller boxes or to avoid clutter
    font_thickness = 1 # Adjusted
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness) # Get text width and height

    text_draw_x = x1
    text_draw_y = y1 - 10

    # Adjust if text goes out of image bounds (top)
    if text_draw_y < text_size[1] + 5: # text_size[1] is height of text
        text_draw_y = y1 + text_size[1] + 10 # Place below the box if no space above

    # Define rectangle background for text
    rect_x1_bg = text_draw_x
    rect_y1_bg = text_draw_y - text_size[1] - 5 # 5 pixels padding above text
    rect_x2_bg = text_draw_x + text_size[0]
    rect_y2_bg = text_draw_y + 5 # 5 pixels padding below text baseline

    cv2.rectangle(yolo_input_img, (rect_x1_bg, rect_y1_bg), (rect_x2_bg, rect_y2_bg), (0, 0, 0), -1) # Black background
    cv2.putText(yolo_input_img, text, (text_draw_x, text_draw_y), font, font_scale, (255, 255, 255), font_thickness) # White text

cv2.imshow('Person Detection with Depth', yolo_input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('person_detection_with_depth.jpg', yolo_input_img)

# Normalize depth_np_resized for visualization (use the one that matches image dimensions)
depth_min = depth_np_resized.min()
depth_max = depth_np_resized.max()
if depth_max - depth_min > 0: # Avoid division by zero
    depth_np_normalized = (depth_np_resized - depth_min) / (depth_max - depth_min)
else:
    depth_np_normalized = np.zeros_like(depth_np_resized)


inv_depth_np_normalized = 1.0 - depth_np_normalized # Invert the normalized values
depth_colormap = cv2.applyColorMap((inv_depth_np_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
cv2.imshow('Inverted Depth Map', depth_colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('inverted_depth_map.jpg', depth_colormap)