import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from tracker import *

# Load YOLO model
model = YOLO('yolov8s.pt')

# Callback function for mouse events
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open video file
cap = cv2.VideoCapture('/Users/abduzohirovmaruf/PycharmProjects/pythonProject23/venv/tf.mp4')

# Read class labels from file
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Counter variables
count = 0
tracker = Tracker()
tracker1 = Tracker()
tracker2 = Tracker()

# Constants for counting lines
cy1 = 184
cy2 = 209
offset = 8

# Dictionaries to store object positions
upcar = {}
downcar = {}
countercarup = []
countercardown = []

upbus = {}
counterbusup = []
downbus = {}
counterbusdown = []

uptruck = {}
countertruckup = []
downtruck = {}
countertruckdown = []

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (1020, 500))

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    # Predict objects using YOLO model
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []
    list1 = []
    list2 = []

    # Extract bounding box coordinates and class labels
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
        elif 'bus' in c:
            list1.append([x1, y1, x2, y2])
        elif 'truck' in c:
            list2.append([x1, y1, x2, y2])

    # Update trackers with bounding box coordinates
    bbox_idx = tracker.update(list)
    bbox1_idx = tracker1.update(list1)
    bbox2_idx = tracker2.update(list2)

    # Process cars
    for bbox in bbox_idx:
        x3, y3, x4, y4, id1 = bbox
        cx3 = int(x3 + x4) // 2
        cy3 = int(y3 + y4) // 2
        if cy1 < (cy3 + offset) and cy1 > (cy3 - offset):
            upcar[id1] = (cx3, cy3)
        if id1 in upcar:
            if cy2 < (cy3 + offset) and cy2 > (cy3 - offset):
                cv2.circle(frame, (cx3, cy3), 4, (255, 0, 0), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)
                if countercarup.count(id1) == 0:
                    countercarup.append(id1)

        if cy2 < (cy3 + offset) and cy2 > (cy3 - offset):
            downcar[id1] = (cx3, cy3)
        if id1 in downcar:
            if cy1 < (cy3 + offset) and cy1 > (cy3 - offset):
                cv2.circle(frame, (cx3, cy3), 4, (255, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
                cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)
                if countercardown.count(id1) == 0:
                    countercardown.append(id1)

    # Process buses
    for bbox1 in bbox1_idx:
        x5, y5, x6, y6, id2 = bbox1
        cx4 = int(x5 + x6) // 2
        cy4 = int(y5 + y6) // 2
        if cy1 < (cy4 + offset) and cy1 > (cy4 - offset):
            upbus[id2] = (cx4, cy4)
        if id2 in upbus:
            if cy2 < (cy4 + offset) and cy2 > (cy4 - offset):
                cv2.circle(frame, (cx4, cy4), 4, (255, 0, 0), -1)
                cv2.rectangle(frame, (x5, y5), (x6, y6), (255, 0, 255), 2)
                cvzone.putTextRect(frame, f'{id2}', (x5, y5), 1, 1)
                if counterbusup.count(id2) == 0:
                    counterbusup.append(id2)

        if cy2 < (cy4 + offset) and cy2 > (cy4 - offset):
            downbus[id2] = (cx4, cy4)
        if id2 in downbus:
            if cy1 < (cy4 + offset) and cy1 > (cy4 - offset):
                cv2.circle(frame, (cx4, cy4), 4, (255, 0, 255), -1)
                cv2.rectangle(frame, (x5, y5), (x6, y6), (255, 0, 0), 2)
                cvzone.putTextRect(frame, f'{id2}', (x5, y5), 1, 1)
                if counterbusdown.count(id2) == 0:
                    counterbusdown.append(id2)

    # Process trucks
    for bbox2 in bbox2_idx:
        x7, y7, x8, y8, id3 = bbox2
        cx5 = int((x7 + x8) / 2)
        cy5 = int((y7 + y8) / 2)

        if cy1 - offset < cy5 < cy1 + offset:
            uptruck[id3] = (cx5, cy5)
        if id3 in uptruck and cy2 - offset < cy5 < cy2 + offset:
            cv2.circle(frame, (cx5, cy5), 4, (0, 255, 0), -1)
            cv2.rectangle(frame, (x7, y7), (x8, y8), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'Truck {id3}', (x7, y7), 1, 1)
            if countertruckup.count(id3) == 0:
                countertruckup.append(id3)

        if cy2 - offset < cy5 < cy2 + offset:
            downtruck[id3] = (cx5, cy5)
        if id3 in downtruck and cy1 - offset < cy5 < cy1 + offset:
            cv2.circle(frame, (cx5, cy5), 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x7, y7), (x8, y8), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'Truck {id3}', (x7, y7), 1, 1)
            if countertruckdown.count(id3) == 0:
                countertruckdown.append(id3)

    # Display counts on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    left_font_scale = 0.8  # Adjust this value to change the size of the left side text
    right_font_scale = 0.8  # Adjust this value to change the size of the right side text
    thickness = 2
    purple_color = (255, 0, 255)  # BGR color for purple
    white_color = (255, 255, 255)  # BGR color for white

    # Left side text moved closer to each other and enclosed in purple boxes
    left_text_up_buses = f'Up Buses: {len(counterbusup)}'
    left_text_down_buses = f'Down Buses: {len(counterbusdown)}'
    left_text_up_trucks = f'Up Trucks: {len(countertruckup)}'
    left_text_down_trucks = f'Down Trucks: {len(countertruckdown)}'

    (left_text_up_buses_width, _), _ = cv2.getTextSize(left_text_up_buses, font, left_font_scale, thickness)
    (left_text_down_buses_width, _), _ = cv2.getTextSize(left_text_down_buses, font, left_font_scale, thickness)
    (left_text_up_trucks_width, _), _ = cv2.getTextSize(left_text_up_trucks, font, left_font_scale, thickness)
    (left_text_down_trucks_width, _), _ = cv2.getTextSize(left_text_down_trucks, font, left_font_scale, thickness)

    # Draw purple boxes and write white text on them
    cv2.rectangle(frame, (50, 60), (50 + left_text_up_buses_width, 30), purple_color, -1)
    cv2.rectangle(frame, (50, 90), (50 + left_text_down_buses_width, 60), purple_color, -1)
    cv2.rectangle(frame, (50, 120), (50 + left_text_up_trucks_width, 90), purple_color, -1)
    cv2.rectangle(frame, (50, 150), (50 + left_text_down_trucks_width, 120), purple_color, -1)

    # Write white text on purple boxes
    cv2.putText(frame, left_text_up_buses, (50, 60), font, left_font_scale, white_color, thickness, cv2.LINE_AA)
    cv2.putText(frame, left_text_down_buses, (50, 90), font, left_font_scale, white_color, thickness, cv2.LINE_AA)
    cv2.putText(frame, left_text_up_trucks, (50, 120), font, left_font_scale, white_color, thickness, cv2.LINE_AA)
    cv2.putText(frame, left_text_down_trucks, (50, 150), font, left_font_scale, white_color, thickness, cv2.LINE_AA)

    # Right side text in purple box with white text
    right_text_up_cars = f'Up Cars: {len(countercarup)}'
    right_text_down_cars = f'Down Cars: {len(countercardown)}'

    (right_text_up_cars_width, _), _ = cv2.getTextSize(right_text_up_cars, font, right_font_scale, thickness)
    (right_text_down_cars_width, _), _ = cv2.getTextSize(right_text_down_cars, font, right_font_scale, thickness)

    # Draw purple boxes and move 'Up Cars' and 'Down Cars' text to the right
    cv2.rectangle(frame, (792, 60), (792 + right_text_up_cars_width, 30), purple_color, -1)
    cv2.rectangle(frame, (792, 160), (792 + right_text_down_cars_width, 130), purple_color, -1)

    # Write white text on purple boxes
    cv2.putText(frame, right_text_up_cars, (792, 60), font, right_font_scale, white_color, thickness, cv2.LINE_AA)
    cv2.putText(frame, right_text_down_cars, (792, 160), font, right_font_scale, white_color, thickness, cv2.LINE_AA)

    # Draw counting lines
    cv2.line(frame, (1, cy1), (1018, cy1), (0, 255, 0), 2)
    cv2.line(frame, (3, cy2), (1016, cy2), (0, 0, 255), 2)

    # Save the frame to the output video
    output_video.write(frame)

    # Display the frame
    cv2.imshow("RGB", frame)

    # Check for the 'Esc' key to exit the loop
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the VideoWriter, release the capture, and destroy all windows
output_video.release()
cap.release()
cv2.destroyAllWindows()
