import ast
import cv2 as cv
import numpy as np
import pandas as pd
from glob import glob
from utils import *

results = pd.read_csv("./outputs/results.csv")

# Read in test video paths and read video by index
videos = glob("./test_sample/*.mp4")
video = cv.VideoCapture(videos[0])

# get video dims
frame_width = int(video.get(3))
frame_height = int(video.get(4))
fps = video.get(cv.CAP_PROP_FPS)
size = (frame_width, frame_height)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('./outputs/processed.mp4', fourcc, fps, size)

# reset video before you re-run cell below
frame_number = -1
video.set(cv.CAP_PROP_POS_FRAMES, 0)

license_plate = {}
for track_id in np.unique(results['track_id']):
    max_ = np.amax(results[results['track_id'] == track_id]['license_text_score'])
    license_plate[track_id] = {'license_crop': None,
                             'license_plate_number': results[(results['track_id'] == track_id) &
                                                             (results['license_text_score'] == max_)]['license_plate_number'].iloc[0]}
    video.set(cv.CAP_PROP_POS_FRAMES, results[(results['track_id'] == track_id) &
                                             (results['license_text_score'] == max_)]['frame_number'].iloc[0])
    ret, frame = video.read()

    x1, y1, x2, y2 = ast.literal_eval(results[(results['track_id'] == track_id) &
                                              (results['license_text_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[track_id]['license_crop'] = license_crop
    
    
ret = True
frame_number = -1
video.set(cv.CAP_PROP_POS_FRAMES, 0)

# read frame
while ret:
    ret, frame = video.read()
    frame_number += 1
    if ret:
        df_ = results[results['frame_number'] == frame_number]
        for row_indx in range(len(df_)):
            # draw car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            
            # draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # crop license plate
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

                cv.putText(frame,
                            license_plate[df_.iloc[row_indx]['track_id']]['license_plate_number'],
                            (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                            cv.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)

            except:
                pass

        out.write(frame)
        frame = cv.resize(frame, (1280, 720))

        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break