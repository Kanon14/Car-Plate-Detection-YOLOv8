import cv2 as cv
from glob import glob
import numpy as np
import pandas as pd
from ultralytics import YOLO

from utils import *

# Initialize the YOLO object detection models for vehicles and license plates
coco_model = YOLO("./yolov8n_train/yolov8n.pt") # Model trained for vehicle detection
np_model = YOLO("./yolov8n_train/best.pt") # Model trained for license plate detection

# Retrieve all video file paths from the specified directory
videos = glob("./test_sample/*.mp4")

results = {}

# Read video by index
video = cv.VideoCapture(videos[0])

ret = True
frame_number = -1
vehicles = [2, 3, 5] # Class IDs for vehicles (e.g., cars, motorcycles, trucks)

# read the entire video
while ret:
    ret, frame = video.read()
    frame_number += 1
    if ret: # Continue if frame is read successfully
        results[frame_number] = {}
        
        # Detect vehicles in the frame using the COCO model
        detections = coco_model.track(frame, persist=True)[0]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score, class_id = detection
            if int(class_id) in vehicles and score > 0.5: # Filter detections by class ID and score
                vehicle_bounding_boxes = []
                vehicle_bounding_boxes.append([x1, y1, x2, y2, track_id, score])
                for bbox in vehicle_bounding_boxes:
                    print(bbox) # Print the bounding box of detected vehicles
                    roi = frame[int(y1):int(y2), int(x1):int(x2)] # Extract region of interest (ROI) for the vehicle
                    
                    # Detect license plates within the ROI using the license plate model
                    license_plates = np_model(roi)[0]
                    for license_plate in license_plates.boxes.data.tolist():
                        plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate
                        # Crop the detected license plate area from the ROI
                        plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                        # Convert license plate to grayscale
                        plate_gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
                        # Apply binary threshold to the grayscale image
                        _, plate_treshold = cv.threshold(plate_gray, 64, 255, cv.THRESH_BINARY_INV)
                        
                        # Apply OCR to the thresholded license plate image
                        np_text, np_score = read_license_plate(plate_treshold)
                        if np_text is not None: # Store the result if the license plate text is successfully read
                            results[frame_number][track_id] = {
                                'car': {
                                    'bbox': [x1, y1, x2, y2],
                                    'bbox_score': score
                                },
                                'license_plate': {
                                    'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                                    'bbox_score': plate_score,
                                    'number': np_text,
                                    'text_score': np_score
                                }
                            }

# Write the accumulated results to a CSV file
write_csv(results, './outputs/results_01.csv')
video.release()

# Load the results from the CSV for analysis
results = pd.read_csv('./outputs/results_01.csv')

# Display results for a specific tracking ID, sorted by OCR prediction confidence
results[results['track_id'] == 1.].sort_values(by='license_text_score', ascending=False)
                