import ast
import cv2 as cv
import easyocr
from glob import glob
import numpy as np
import pandas as pd
import string
from ultralytics import YOLO

# Initiate the detection model
coco_model = YOLO("./yolov8n_train/yolov8n.pt") # For cars
np_model = YOLO("./yolov8n_train/best.pt") # For car plate

# Read in test video paths
videos = glob("./test_sample/*.mp4")

# read video by index
video = cv.VideoCapture(videos[3])

ret = True
frame_number = -1
vehicles = [2,3,5]

# read the 10 first frames
while ret:
    frame_number += 1
    ret, frame = video.read()

    if ret and frame_number < 10:
        
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
                    # debugging check if bbox lines up with detected vehicles (should be identical to save_crops() above
                    # cv.imwrite(str(track_id) + '.jpg', roi)
                    
                    # license plate detector for region of interest
                    license_plates = np_model(roi)[0]
                    # check every bounding box for a license plate
                    for license_plate in license_plates.boxes.data.tolist():
                        plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate
                        # verify detections
                        print(license_plate, 'track_id: ' + str(bbox[4]))
                        plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                        cv.imwrite(str(track_id) + '.jpg', plate)
                        
video.release()
                