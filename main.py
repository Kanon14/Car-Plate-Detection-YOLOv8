import cv2 as cv
from glob import glob
import numpy as np
import pandas as pd
from ultralytics import YOLO

from utils import *

# Initiate the detection model
coco_model = YOLO("./yolov8n_train/yolov8n.pt") # For cars
np_model = YOLO("./yolov8n_train/best.pt") # For car plate

# Read in test video paths
videos = glob("./test_sample/*.mp4")

results = {}

# Read video by index
video = cv.VideoCapture(videos[0])

ret = True
frame_number = -1
vehicles = [2,3,5]

# read the entire video
while ret:
    ret, frame = video.read()
    frame_number += 1
    if ret: # Testing with 100 frames first
        results[frame_number] = {}
        
        # vehicle detector
        detections = coco_model.track(frame, persist=True)[0]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score, class_id = detection
            if int(class_id) in vehicles and score > 0.5:
                vehicle_bounding_boxes = []
                vehicle_bounding_boxes.append([x1, y1, x2, y2, track_id, score])
                for bbox in vehicle_bounding_boxes:
                    print(bbox)
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    # license plate detector for region of interest
                    license_plates = np_model(roi)[0]
                    # process license plate
                    for license_plate in license_plates.boxes.data.tolist():
                        plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate
                        # crop plate from region of interest
                        plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                        # de-colorize
                        plate_gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
                        # posterize
                        _, plate_treshold = cv.threshold(plate_gray, 64, 255, cv.THRESH_BINARY_INV)
                        
                        # OCR
                        np_text, np_score = read_license_plate(plate_treshold)
                        # if plate could be read write results
                        if np_text is not None:
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

write_csv(results, './results.csv')
video.release()

results = pd.read_csv('./results.csv')

# show results for tracking ID `1` - sort by OCR prediction confidence
results[results['track_id'] == 1.].sort_values(by='license_text_score', ascending=False)
                