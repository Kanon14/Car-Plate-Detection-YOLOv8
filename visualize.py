import ast
import cv2 as cv
import numpy as np
import pandas as pd
from glob import glob
from utils import *

# Load results from a CSV containing vehicle and license plate detection information
results = pd.read_csv("./outputs/results_01.csv")

# Retrieve paths to video files and open the first video
videos = glob("./test_sample/*.mp4")
video = cv.VideoCapture(videos[0])

# Get dimensions and frame rate of the video
frame_width = int(video.get(3))
frame_height = int(video.get(4))
fps = video.get(cv.CAP_PROP_FPS)
size = (frame_width, frame_height)

# Setup the video writer object to save output video with annotations
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('./outputs/processed_01.mp4', fourcc, fps, size)

# Reset video before you re-run cell below
frame_number = -1
video.set(cv.CAP_PROP_POS_FRAMES, 0)

# Read and process the highest score license plate for each tracked vehicle
license_plate = {}
for track_id in np.unique(results['track_id']):
    # Find the frame with the highest OCR confidence for each track_id
    max_ = np.amax(results[results['track_id'] == track_id]['license_text_score'])
    # Get the corresponding license plate number and crop the image of that plate
    license_plate[track_id] = {'license_crop': None,
                             'license_plate_number': results[(results['track_id'] == track_id) &
                                                             (results['license_text_score'] == max_)]['license_plate_number'].iloc[0]}
    video.set(cv.CAP_PROP_POS_FRAMES, results[(results['track_id'] == track_id) &
                                             (results['license_text_score'] == max_)]['frame_number'].iloc[0])
    ret, frame = video.read()
    # Calculate the coordinates for the license plate
    x1, y1, x2, y2 = ast.literal_eval(results[(results['track_id'] == track_id) &
                                              (results['license_text_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[track_id]['license_crop'] = license_crop
    
# Reset video to start frame
ret = True
frame_number = -1
video.set(cv.CAP_PROP_POS_FRAMES, 0)

# Annotate each frame with vehicle and license plate information
while ret:
    ret, frame = video.read()
    frame_number += 1
    if ret: # Get data for the current frame
        df_ = results[results['frame_number'] == frame_number]
        for row_indx in range(len(df_)):
            # Draw bounding box around the car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            
            # Draw bounding box around the license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # Add cropped license plate image to frame
            license_crop = license_plate[df_.iloc[row_indx]['track_id']]['license_crop']

            H, W, _ = license_crop.shape

            try:
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                (text_width, text_height), _ = cv.getTextSize(
                    license_plate[df_.iloc[row_indx]['track_id']]['license_plate_number'],
                    cv.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)

                # Add license plate number text above the cropped image
                cv.putText(frame,
                            license_plate[df_.iloc[row_indx]['track_id']]['license_plate_number'],
                            (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                            cv.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)

            except Exception as e:
                print("Error placing license plate image or text:", e)

        # Write the annotated frame to the output video
        out.write(frame)
        frame = cv.resize(frame, (1280, 720))

        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break