import cv2
from ultralytics import YOLO
from depth_estimation import estimate_depth_for_object
from utils import get_package_with_minimum_depth

class_list = ["label", "package"]

# Generate random colors for class list
detection_colors = [(38, 76, 227), (0, 255, 0), ]


def detect_and_crop_package(frame):
    # Load the trained YOLO model
    model = YOLO("best.pt", "v8")

    package_data = []

    # Initialize the depth map for the entire scene
    depth_map = None

    # Predict on the frame
    detect_params = model.predict(source=[frame], conf=0.8, save=False)

    # If depth map is not available, calculate it
    if depth_map is None:
        depth_map = estimate_depth_for_object(frame)

    # Iterate over detected objects and draw bounding boxes and circles on the snapshot
    if len(detect_params) > 0:
        for i, box in enumerate(detect_params[0].boxes):
            cls_id = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            x, y = int(bb[0]), int(bb[1])
            w, h = int(bb[2]) - x, int(bb[3]) - y

            if class_list[int(cls_id)] == "package":
                # Calculate the center coordinates of the detected package
                center_x = x + (w // 2)
                center_y = y + (h // 2)

                # Calculate depth data for the detected package using the precomputed depth map
                package_depth = depth_map[y:y + h, x:x + w]

                # Check if there is any label inside the package
                has_label = False

                # Iterate over detected objects again to find labels inside the package
                for j, inner_box in enumerate(detect_params[0].boxes):
                    inner_cls_id = inner_box.cls.numpy()[0]
                    inner_bb = inner_box.xyxy.numpy()[0]

                    inner_x, inner_y = int(inner_bb[0]), int(inner_bb[1])

                    # Check if the inner object is a label and is contained within the package
                    if class_list[int(inner_cls_id)] == "label" and x <= inner_x <= (x + w) and y <= inner_y <= (y + h):
                        has_label = True
                        break

                # Create a dictionary for package data
                package_info = {
                    "class": class_list[int(cls_id)],
                    "confidence": conf,
                    "depthData": package_depth.mean(),
                    "center": (center_x, center_y),
                    "width": w,
                    "height": h,
                    "coordinates": (x, y),
                    "has_label": has_label,  # Add has_label field
                }

                # Append the package data to the list
                package_data.append(package_info)

    min_depth_package = get_package_with_minimum_depth(package_data)
    cropped_package = None

    if min_depth_package is not None:
        # Get the coordinates and dimensions of the selected package
        x, y = min_depth_package['coordinates']
        width = min_depth_package['width']
        height = min_depth_package['height']

        # Crop the selected package from the frame
        cropped_package = frame[y:y + height, x:x + width]

        # Display the "Final Depth Map" and "Object Detection" windows
        cv2.imshow("Final Depth Map", depth_map)

        # Display class name and confidence
        font = cv2.FONT_HERSHEY_SIMPLEX  # Change the font here

        # Determine the text and rectangle colors based on 'has_label'
        if min_depth_package["has_label"]:
            text_color = (0, 0, 0)  # Green color for text
            rectangle_color = (0, 255, 0)  # Green color for bounding box
        else:
            text_color = (0, 0, 0)  # Red color for text
            rectangle_color = (0, 0, 255)  # Red color for bounding box

        # Calculate the background rectangle position and size for the text
        text_bg_x1 = x - 1
        text_bg_x2 = x + width + 1
        text_bg_y1 = y - 32  # Adjust the background rectangle position
        text_bg_y2 = y - 0  # Adjust the background rectangle position

        # Draw a filled background rectangle for the text
        cv2.rectangle(
            frame,
            (text_bg_x1, text_bg_y1),
            (text_bg_x2, text_bg_y2),
            rectangle_color,  # Use the determined rectangle color
            thickness=cv2.FILLED,  # Fill the rectangle
        )

        # Draw a rectangle around the detected object on the snapshot
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            rectangle_color,  # Use the determined rectangle color
            thickness=2,  # Increase the thickness to 2
        )

        # Draw the text inside the background rectangle
        cv2.putText(
            frame,
            "Package " + str(round(min_depth_package["confidence"], 3)) + "%",
            (x+10, y - 10),  # Adjust the text position to be inside the background rectangle
            font,
            0.5,
            text_color,  # Use the determined text color
            1,
        )

        cv2.circle(frame,  min_depth_package["center"], 5, (255, 0, 0), -1)

        cv2.imshow("Object Detection", frame)

    return depth_map,frame,cropped_package, min_depth_package  # Return the cropped package and package info
