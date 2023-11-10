import cv2
import random
import numpy as np
import torch
import os

from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("best.pt", "v8")

class_list = ["label", "package"]

# Generate random colors for class list
detection_colors = [(38, 76, 227), (208, 6, 107)]

# Create a video capture object to capture video from your camera (0 indicates the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Predict on the frame
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    if len(detect_params) > 0:
        topmost_box = None  # Initialize the topmost box
        for i, box in enumerate(detect_params[0].boxes):
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            x, y = int(bb[0]), int(bb[1])
            w, h = int(bb[2]) - x, int(bb[3]) - y

            if class_list[int(clsID)] == "package":
                # Calculate the center coordinates of the detected package
                center_x = x + (w // 2)
                center_y = y + (h // 2)

                # Draw a rectangle around the detected object
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    detection_colors[int(clsID)],
                    2,
                )

                # Create a circle only for the "package" class
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    class_list[int(clsID)] + " " + str(round(conf, 3)) + "%" + f"({x},{y})",
                    # Added '+' after "%" and comma after the string
                    (x, y - 10),  # Added a comma here to separate the position tuple
                    font,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # Print the data of the detected package
                print(f"Package Data - Class: {class_list[int(clsID)]}, Confidence: {round(conf, 3)}%, "
                      f"Center Coordinates: ({center_x}, {center_y})")

                # Check if this box is the topmost one
                if topmost_box is None or y < topmost_box[1]:
                    topmost_box = (x, y, x + w, y + h)

        # Draw a rectangle around the topmost box (if found)
        if topmost_box is not None:
            cv2.rectangle(
                frame,
                (topmost_box[0], topmost_box[1]),
                (topmost_box[2], topmost_box[3]),
                (0, 255, 0),  # Green color for the topmost box
                2,
            )

    # Display the frame with object detection in one window
    cv2.imshow("ObjectDetection", frame)

    # Check for a key press and exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()









#=========================STATIC======================================

# import cv2
# import random
# import numpy as np
# import torch
# import os
#
# from ultralytics import YOLO
#
# # Load the trained YOLO model
# model = YOLO("best.pt", "v8")
#
# # Load the static image
# image = cv2.imread("4.png")  # Replace "your_static_image.jpg" with the path to your image
#
# class_list = ["label", "package"]
#
# # Generate random colors for class list
# detection_colors = [(38, 76, 227), (208, 6, 107)]
#
# # Predict on image
# detect_params = model.predict(source=[image], conf=0.45, save=False)
#
# if len(detect_params) > 0:
#     topmost_box = None  # Initialize the topmost box
#     for i, box in enumerate(detect_params[0].boxes):
#         clsID = box.cls.numpy()[0]
#         conf = box.conf.numpy()[0]
#         bb = box.xyxy.numpy()[0]
#
#         x, y = int(bb[0]), int(bb[1])
#         w, h = int(bb[2]) - x, int(bb[3]) - y
#
#         if class_list[int(clsID)] == "package":
#             # Calculate the center coordinates of the detected package
#             center_x = x + (w // 2)
#             center_y = y + (h // 2)
#
#             # Draw a rectangle around the detected object
#             cv2.rectangle(
#                 image,
#                 (x, y),
#                 (x + w, y + h),
#                 detection_colors[int(clsID)],
#                 2,
#             )
#
#             # Create a circle only for the "package" class
#             cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)
#
#             # Display class name and confidence
#             font = cv2.FONT_HERSHEY_COMPLEX
#             cv2.putText(
#                 image,
#                 class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
#                 (x, y - 10),
#                 font,
#                 0.5,
#                 (255, 255, 255),
#                 1,
#             )
#
#             # Print the data of the detected package
#             print(f"Package Data - Class: {class_list[int(clsID)]}, Confidence: {round(conf, 3)}%, "
#                   f"Center Coordinates: ({center_x}, {center_y})")
#
#             # Check if this box is the topmost one
#             if topmost_box is None or y < topmost_box[1]:
#                 topmost_box = (x, y, x + w, y + h)
#
#     # Draw a rectangle around the topmost box (if found)
#     if topmost_box is not None:
#         cv2.rectangle(
#             image,
#             (topmost_box[0], topmost_box[1]),
#             (topmost_box[2], topmost_box[3]),
#             (0, 255, 0),  # Green color for the topmost box
#             2,
#         )
#
# # Display the resulting image
# cv2.imshow("ObjectDetection", image)
#
# # Wait for a key press and then close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()
