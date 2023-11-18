"""
The below code takes an image filename as i/p in the variable filename and background environment filename as i/p in variable filename_env
It generates an image : env_containing_window with the window as background in the environment
It generates the mask of the window : closing2
Both env_containing_window and closing2 can be used as inputs and groundtruths so as to train window detection models
"""


import cv2
import numpy as np
from PIL import Image, ImageFilter
import random
import math
import os


envi_list = ['andromeda.jpeg', 'donuts.jpg',
             'tides.jpeg', 'lavender_nature.jpeg', 'halloween.jpg']

curr_folder = os.path.abspath(__file__)
curr_folder = curr_folder.replace("image_warp.py","")
image_path = curr_folder + str('/real_world_imgs') #'/home/ssuryalolla/Aerial_Robotics/data_augment/real_world_imgs'
envi_path = curr_folder + str('/environments') # '/home/ssuryalolla/Aerial_Robotics/data_augment/environments'
mask_path = curr_folder + str('/real_world_masks') #'/home/ssuryalolla/Aerial_Robotics/data_augment/real_world_masks'

augmented_image_dir = curr_folder + str('/aug_real_world_imgs')#'/home/ssuryalolla/Aerial_Robotics/data_augment/aug_real_world_imgs'
augmented_mask_dir =  curr_folder + str('/aug_real_world_masks')#'/home/ssuryalolla/Aerial_Robotics/data_augment/aug_real_world_masks'

if not os.path.exists(augmented_image_dir):
    os.makedirs(augmented_image_dir)

if not os.path.exists(augmented_mask_dir):
    os.makedirs(augmented_mask_dir)

image_files = sorted(os.listdir(image_path))
mask_files = sorted(os.listdir(mask_path))

for k in range(len(image_files)):
    filename = image_files[k]
    filename_mask = mask_files[k]

    for j in range(300):

        full_image_path = os.path.join(image_path, filename)

        filename_env = envi_list[random.randint(0, len(envi_list) - 1)]
        full_envi_path = os.path.join(envi_path, filename_env)

        full_mask_path = os.path.join(mask_path, filename_mask)

        img = cv2.imread(full_image_path)
        # let's downscale the image using new  width and height
        env = cv2.imread(full_envi_path)
        label_mask_actual = cv2.imread(full_mask_path)

        down_width = 480
        down_height = 360
        down_points = (down_width, down_height)
        img = cv2.resize(img, down_points, interpolation=cv2.INTER_LINEAR)
        resized_down_env = cv2.resize(
            env, down_points, interpolation=cv2.INTER_LINEAR)
        label_mask = cv2.resize(
            label_mask_actual, down_points, interpolation=cv2.INTER_LINEAR)

        gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thrash = cv2.threshold(gray, 70, 255, cv2.CHAIN_APPROX_NONE)

        # Find the contours in the thresholded image
        contours, hierarchy = cv2.findContours(
            thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours by area and get the largest contour
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Assuming the largest contour is the outermost square, get it
        outermost_square = sorted_contours[0]

        # Create a blank mask of the same size as the original image
        mask = np.zeros_like(gray)

        # Draw the outermost square on the mask
        cv2.drawContours(mask, [outermost_square], -1,
                         (255), thickness=cv2.FILLED)

        # To fill inside the inner square with white, use the second largest contour (assuming it's the inner square)
        inner_square = sorted_contours[1]
        cv2.drawContours(mask, [inner_square], -1, (255), thickness=cv2.FILLED)

        # Combine the mask with the original image
        im_out = cv2.bitwise_or(img, img, mask=mask)

        # cv2.imshow('foreground', im_out)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Define a kernel for the morphological operation (change the size as needed)
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply opening operation
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find the bounding rectangle
        x, y, w, h = cv2.boundingRect(opened)

        # Create a new clean square mask
        mask1 = np.zeros_like(mask)
        cv2.rectangle(mask1, (x, y), (x+w, y+h), 255, -1)

        # cv2.drawContours(mask, [contour], 0, (0,255,0), 3)
        # cv2.imshow('smooth mask', mask1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        contours, _ = cv2.findContours(
            mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)

        # drawing points

        pts1 = []
        pts2 = []

        for i, point in enumerate(approx):
            x, y = point[0]
            # print(x,y)
            # cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            # Generating a random point inside the circle
            angle = 2 * math.pi * random.random()  # Random angle between 0 and 2π
            if i == 0:
                # Random distance from center
                distance = random.randint(20, 50) * math.sqrt(random.random())
            if i == 1:
                distance = random.randint(20, 50) * math.sqrt(random.random())
            if i == 2:
                distance = random.randint(20, 50) * math.sqrt(random.random())
            if i == 3:
                distance = random.randint(20, 50) * math.sqrt(random.random())
            a = int(x + distance * math.cos(angle))
            b = int(y + distance * math.sin(angle))

            pts1.append([x, y])
            pts2.append([a, b])

        starts = np.float32(np.array(pts1))
        warped = np.float32(np.array(pts2))

        # Apply Perspective Transform Algorithm
        matrix = cv2.getPerspectiveTransform(starts, warped)
        result = cv2.warpPerspective(img, matrix, (480, 360))

        warped_mask = cv2.warpPerspective(label_mask, matrix, (480, 360))

        # Convert to grayscale
        gray_mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)

        # Threshold to create a binary mask
        _, warped_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours_mask, _ = cv2.findContours(
            warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If there are more than 2 contours, it means there are other unwanted elements in the image.
        # In this case, sort the contours by area and keep the two largest ones (the concentric squares).
        if len(contours_mask) > 2:
            contours_mask = sorted(
                contours_mask, key=cv2.contourArea, reverse=True)[:2]

        # Draw the contours, filling the space between them with white
        # -1 thickness means fill
        cv2.drawContours(warped_mask, contours_mask, -1,
                         (255, 255, 255), thickness=-1)

        gray_result = 255 - cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thrash_result = cv2.threshold(
            gray_result, 65, 255, cv2.CHAIN_APPROX_NONE)

        kernel2 = np.ones((30, 30), np.uint8)
        closing2 = cv2.morphologyEx(thrash_result, cv2.MORPH_CLOSE, kernel2)
        inv_closing2 = cv2.bitwise_not(closing2)

        result_withoutbg = cv2.bitwise_and(result, result, mask=closing2)

        env_through_window = cv2.bitwise_and(
            resized_down_env, resized_down_env, mask=closing2)

        env_without_window_mask = cv2.bitwise_not(env_through_window)

        # env_without_window = np.where(inv_closing2[:, :, None].astype(bool), resized_down_env, 0)
        env_without_window = cv2.bitwise_and(
            resized_down_env, env_without_window_mask)

        # cv2.imshow('window without bg', warped_mask)
        # cv2.waitKey(0)

        # cv2.imshow('without window mask', inv_closing2) # Transformed Capture
        # cv2.imshow('env w/o window', env_without_window) # Transformed Capture

        env_containing_window = cv2.add(result_withoutbg, env_without_window)

        # Generate unique filenames
        filename_augimage = f"augFrame{k:04d}{j:04d}.png"
        filename_augmask = f"augFrame{k:04d}{j:04d}.png"

        aug_image_path = os.path.join(augmented_image_dir, filename_augimage)
        aug_mask_path = os.path.join(augmented_mask_dir, filename_augmask)

        # Save the images
        cv2.imwrite(aug_image_path, env_containing_window)
        cv2.imwrite(aug_mask_path, warped_mask)

        # cv2.imshow('env containing window', env_containing_window)
        # cv2.imshow('label mask', warped_mask)

        # cv2.waitKey(0)

        # cv2.destroyAllWindows()


# kernel = np.ones((30, 30), np.uint8)
# closing = cv2.morphologyEx(thrash, cv2.MORPH_CLOSE, kernel)
# edges = cv2.Canny(closing, 100, 200)

# dst = cv2.cornerHarris(im_out,2,3,0.04)
# dst = cv2.dilate(dst,None)
# img[dst>0.01*dst.max()] = [0,0,255]
