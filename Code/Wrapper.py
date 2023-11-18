import os
import time
import sys

import cv2
import numpy as np

from djitellopy import Tello
from ultralytics import YOLO
from threading import Thread

# ----------pnp visualizations-----------------------------------


def draw_axis(img, rotvec, tvec, K):
    # unit is mm
    dist_coeffs = np.zeros((5, 1))
    dist_coeffs[0][0] = 0.02456386593401987
    dist_coeffs[1][0] = -0.5958069654037562
    dist_coeffs[2][0] = -0.0003932676388405013
    dist_coeffs[3][0] = -0.00017064279541258975
    dist_coeffs[4][0] = 1.8486532081847153
    points = np.float32(
        [[20, 0, 0], [0, 20, 0], [0, 0, 20], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotvec, tvec, K, dist_coeffs)
    axisPoints = axisPoints.astype(int)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[0].ravel()), (255, 0, 0), 3)  # Blue is x axis
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[1].ravel()), (0, 255, 0), 3)  # Green is Y axis
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[2].ravel()), (0, 0, 255), 3)  # Red is z axis
    return img


# ------Corner infer funcs--------------------------------------
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def get_corners(img, mask):
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((3, 3), np.uint8)

    # The first parameter is the original image,
    # kernel is the matrix with which image is convolved and third parameter is the number
    # iterations will determine how much you want to erode/dilate a given image.
    img_erosion = cv2.erode(mask, kernel, iterations=15)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=15)

    img_resized = resize_with_aspect_ratio(img, width=960)
    resized_image = resize_with_aspect_ratio(
        mask, width=960)  # You can adjust the width as needed
    resized_image_erosion = resize_with_aspect_ratio(
        img_erosion, width=960)  # You can adjust the width as needed
    resized_image_dilation = resize_with_aspect_ratio(
        img_dilation, width=960)  # You can adjust the width as needed

    contours, _ = cv2.findContours(
        np.uint8(resized_image_dilation), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    resized_image_dilation_color = cv2.cvtColor(
        resized_image_dilation, cv2.COLOR_GRAY2BGR)

    corners = []
    for contour in contours:
        # Approximate polygon and ensure it has 4 corners
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            # Draw circles on the corner points
            for point in approx:
                x, y = point[0]
                cv2.circle(img_resized, (int(x), int(y)), 7, (0, 0, 255), -1)
                corners.append((x, y))

    # cv2.imshow('Input', resized_image)
    # cv2.imshow('Erosion', resized_image_erosion)
    # cv2.imshow('Dilation', resized_image_dilation_color)
    # cv2.imshow('Detected contours', img)

    corner_filename = current_path + str("/outputs/corners/") + f"frame{i}.png"
    cv2.imwrite(corner_filename, img_resized)

    return corners, img_resized


def order_points(points):
    # Sort points by x-coordinate (leftmost will be top-left, rightmost will be top-right)
    sorted_points = sorted(points, key=lambda x: x[0])
    print("sorted func output", sorted_points)

    # left most and right most points
    left1 = sorted_points[0]
    left2 = sorted_points[1]

    right1 = sorted_points[-1]
    right2 = sorted_points[-2]

    if left1[1] > left2[1]:
        bottom_left = left1
        top_left = left2
    else:
        bottom_left = left2
        top_left = left1

    if right1[1] > right2[1]:
        top_right = right2
        bottom_right = right1
    else:
        top_right = right1
        bottom_right = right2
    # # Identify top-left and top-right points
    # top_left = sorted_points[0]
    # top_right = sorted_points[1]

    # Calculate the distance between the top-left and top-right points
    # The point with smaller y-coordinate is top-left, the other is top-right
    # if top_left[1] > top_right[1]:
    #     top_left, top_right = top_right, top_left

    # # Identify bottom-left and bottom-right points
    # bottom_left = sorted_points[2] if sorted_points[2][1] > sorted_points[3][1] else sorted_points[3]
    # bottom_right = sorted_points[3] if sorted_points[2][1] > sorted_points[3][1] else sorted_points[2]

    return [top_left, bottom_left, bottom_right, top_right]


def get_axis(img, corners):
    K = np.array([[917.3527180617801, 0.0, 480.97134568905716], [
                 0.0, 917.1043451426654, 365.57078783755276], [0.0, 0.0, 1.0]])
    points_2D = np.array([corners], dtype="double")
    points_3D = np.array([
                        (-50.8, 45.72, 0),     # First
                        (-50.8, -45.72, 0),  # Second
                        (50.8, -45.72, 0),  # Third
                        (50.8, 45.72, 0)  # Fourth
    ])
    dist_coeffs = np.zeros((5, 1))
    dist_coeffs[0][0] = 0.02456386593401987
    dist_coeffs[1][0] = -0.5958069654037562
    dist_coeffs[2][0] = -0.0003932676388405013
    dist_coeffs[3][0] = -0.00017064279541258975
    dist_coeffs[4][0] = 1.8486532081847153
    success, rotation_vector, translation_vector = cv2.solvePnP(
        points_3D, points_2D, K, dist_coeffs, flags=0)
    image = draw_axis(img, rotation_vector, translation_vector, K)
    axis_filename = current_path + str("/outputs/axes/") + f"frame{i}.png"
    cv2.imwrite(axis_filename, image)
    return rotation_vector, translation_vector


def recordWorker():
    j = 0
    while True:
        frame = drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        filename = current_path + \
            str("/outputs/frames_thread/") + f"frame{j}.png"
        cv2.imwrite(filename, frame)
        j = j+1


# Read env file [ax ay az] [bx by bz] [cx cy cz]..
# Initialize a list to store the window coordinates
window_coordinates = []

# Read the file line by line
with open('Environment/env1.txt', 'r') as file:
    for line in file:
        if line.startswith('#'):
            continue  # Skip comments
        elif line.startswith('window'):
            parts = line.split()  # Split the line into parts
            x, y, z = int(float(parts[1]) * 100), int(float(parts[2])
                                                      * 100), int(float(parts[3]) * 100)  # Extract x, y, z
            window_coordinates.append((x, y, z))  # Store them as a tuple

# Convert the coordinates to the desired format
formatted_coordinates = [(x, y, z) for x, y, z in window_coordinates]

# Assign the variables ax, ay, az, bx, by, bz, etc.
ax, ay, az = formatted_coordinates[0]
bx, by, bz = formatted_coordinates[1]
cx, cy, cz = formatted_coordinates[2]

# Get the directory of the script being run
current_path = os.path.dirname(os.path.abspath(__file__))
current_path = current_path.replace("Wrapper.py", "")
print(current_path)

# Define the 'output' directory path
output_directory = os.path.join(current_path, "outputs")

# Define the directories to create within the 'output' directory
frame_directory = os.path.join(output_directory, "frames")
corners_directory = os.path.join(output_directory, "corners")
masks_directory = os.path.join(output_directory, "masks")
axes_directory = os.path.join(output_directory, "axes")
frames_thread_directory = os.path.join(output_directory, "frames_thread")

# List of directories to be created
directories = [
    frame_directory,
    corners_directory,
    masks_directory,
    axes_directory,
    frames_thread_directory
]

# First, create the 'output' directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Then, loop through the directories list and create each one if it doesn't exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

try:
    current_path = os.path.abspath(__file__)
    current_path = current_path.replace("Wrapper.py", "")
    model_path = current_path + \
        str("YOLO Model/runs/segment/train2/weights/last.pt")

    model = YOLO(model_path)

    # Tello drone code template:
    # Check the connection to the drone
    drone = Tello()
    drone.connect()

    # CHECK SENSOR READINGS------------------------------------------------------------------
    print('Altitude ', drone.get_distance_tof())
    print('Battery, ', drone.get_battery())

    center = []
    drone.streamon()
    drone.takeoff()

    # create and start the movement thread
    Thread(target=recordWorker).start()

    # Go to a initial location
    speed = 45
    counter = 0
    i = 0
    wp_list = [(0, -ax, 80), (0, ax-bx, 0), (0, bx-cx, 0)]
    depth_add = 40  # tunable
    groundtruth_list = [(ay + depth_add, 0, 0), (by - ay +
                                                 depth_add, 0, 0), (cy - by + depth_add, 0, 0)]
    while counter < 3:
        time.sleep(2)
        # Send command to predefined waypoint
        drone.go_xyz_speed(
            wp_list[counter][0], wp_list[counter][1], wp_list[counter][2], speed)
        while True:
            frame = drone.get_frame_read().frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            H, W, _ = frame.shape
            print("shape", (H, W))
            filename = current_path + str("/outputs/frames") + f"frame{i}.png"
            cv2.imwrite(filename, frame)
            # Running the model on the saved frame
            # # To run the model
            # results = model(img)
            # mask = results[0].masks.data
            # mask = mask.cpu().numpy()*255
            # mask = cv2.resize(mask[0], (W, H))
            # cv2.imwrite('./output.png', mask)

            # # To get corners
            # corner_img = get_corners(img,mask)
            frame = cv2.imread(filename)
            results = model(frame)
            try:
                mask = results[0].masks.data
                mask = mask.cpu().numpy()*255
                mask = cv2.resize(mask[0], (W, H))
                mask_filename = current_path + \
                    str("/outputs/masks/") + f"frame{i}.png"
                cv2.imwrite(mask_filename, mask)

                # To get corners
                corners, corner_img = get_corners(frame, mask)
                # Reorder the corners
                corners = order_points(corners)
                rvec, tvec = get_axis(corner_img, corners)
                if True:
                    drone.go_xyz_speed(int((tvec[2][0]+groundtruth_list[counter][0])/2), int(
                        (tvec[1][0] + 0)/2), int((tvec[0][0]+0)/2)-10, speed)
                    i += 1
                    break
            except:
                continue
        counter += 1
    # drone.land()

except KeyboardInterrupt:
    drone.land()
    # HANDLE KEYBOARD INTERRUPT AND STOP THE DRONE COMMANDS
    print('keyboard interrupt')
    drone.emergency()
    drone.emergency()
    drone.end()
