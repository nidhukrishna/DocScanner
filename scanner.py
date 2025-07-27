from ultralytics import YOLO
import cv2
import numpy as np


#Loading model
model_path = r"C:.\best.pt"
model = YOLO(model_path)

#Loading Image
image_path = r"C:./sample/d2.jpg"
image = cv2.imread(image_path)


#Predict Result
results = model(image)

#Extract Result
for r in results:
    # Check if any bounding boxes were detected
    if r.boxes:
        
        box = r.boxes[0]
        
        # Get the coordinates in (x1, y1, x2, y2) format
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        
        # Crop the image using the coordinates
        cropped_image = image[y1:y2, x1:x2]
        
        # Original image with detection
        annotated_image = r.plot()
        
        
        
        
        cv2.imshow('YOLOv8 Detection', annotated_image)
        
        cv2.imshow('Cropped Document', cropped_image)
        
        
        
        cv2.waitKey(0)

cv2.destroyAllWindows()