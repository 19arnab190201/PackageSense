import cv2
import numpy as np

def show_dashboard():
    # Load your images
    image1 = cv2.imread('image1.png')
    image2 = cv2.imread('image2.png')
    image3 = cv2.imread('image3.png')
    image4 = cv2.imread('image4.png')

    # Resize images if necessary to make them the same size
    # (assuming all images have the same dimensions)
    image1 = cv2.resize(image1, (300, 200))
    image2 = cv2.resize(image2, (300, 200))
    image3 = cv2.resize(image3, (300, 200))
    image4 = cv2.resize(image4, (300, 200))

    # Create an empty canvas to display the images
    canvas = np.zeros((800, 1200, 3), dtype=np.uint8)

    # Define the positions and sizes for different sections
    left_width = int(canvas.shape[1] * 0.7)
    right_width = int(canvas.shape[1] * 0.3)
    top_height = int(canvas.shape[0] * 0.5)
    bottom_height = int(canvas.shape[0] * 0.5)

    # Create the left-top section with 2 rows and 2 image placeholders
    left_top = canvas[0:top_height, 0:left_width]
    left_top[0:200, 0:300] = image1
    left_top[200:, 0:300] = image2

    # Create the left-bottom section with 2 rows and 2 image placeholders
    left_bottom = canvas[top_height:, 0:left_width]
    left_bottom[0:200, 0:300] = image3
    left_bottom[200:, 0:300] = image4

    # Create the right section with 3 rows
    right = canvas[:, left_width:]
    title_box = np.zeros((100, right.shape[1], 3), dtype=np.uint8)
    packet_count = np.zeros((100, right.shape[1], 3), dtype=np.uint8)
    flapper_status = np.zeros((100, right.shape[1], 3), dtype=np.uint8)
    coordinates = np.zeros((500, right.shape[1], 3), dtype=np.uint8)

    # Set titles and text in the right section
    cv2.putText(title_box, "System Status", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(packet_count, "Packet Count: 0", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(flapper_status, "Flapper Status: ON", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(coordinates, "X: 0.00  Y: 0.00  Z: 0.00", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Place the sections in the right section
    right[0:100] = title_box
    right[100:200] = packet_count
    right[200:300] = flapper_status
    right[300:] = coordinates

    # Display the canvas with all sections in a single window
    cv2.imshow('Dashboard', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()