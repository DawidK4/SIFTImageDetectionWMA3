import cv2
import numpy as np
import glob
import os

# zad 2

key_images_path = "photos/"

key_images = [] # List to store images, keypoints, and descriptors
key_descriptors = [] # List to store descriptors for later use
sift = cv2.SIFT_create() # Initialize SIFT Detector

for file in sorted(glob.glob(os.path.join(key_images_path, "*.jpg"))):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE) # Read image in grayscale
    mask = np.ones_like(image, dtype=np.uint8) * 255 # Create a mask for the image

    keypoints, descriptors = sift.detectAndCompute(image, mask) # Detect keypoints and compute descriptors
    key_images.append((image, keypoints, descriptors, file))   # Store image, keypoints, descriptors, and filename 
    key_descriptors.append(descriptors) # Store descriptors for later use

    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None) # Draw keypoints on the image
    cv2.imshow(f"Keypoints {file}", img_with_keypoints)
    cv2.waitKey(500)

cv2.destroyAllWindows()

# zad 3

video = cv2.VideoCapture("video.mp4") # Open video file
bf = cv2.BFMatcher() # Initialize Brute Force Matcher

frame_count = 0 
match_counts = {img[3]: 0 for img in key_images} # Dictionary to count matches for each key image

while video.isOpened():
    ret, frame = video.read() 
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None) # Detect keypoints and compute descriptors for the current frame

    best_match_img = None 
    max_matches = 0
    best_matches = None

    for img, kp, des, filename in key_images:
        if des is None or des_frame is None:
            continue
        matches = bf.knnMatch(des, des_frame, k=2) # Find the two best matches for each descriptor

        good_matches = [] 
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > max_matches:
            max_matches = len(good_matches)
            best_match_img = img
            best_matches = good_matches
            best_filename = filename

    if best_match_img is not None:
        match_counts[best_filename] += 1
        img_matches = cv2.drawMatches(best_match_img, key_images[0][1], frame, kp_frame, best_matches, None)
        cv2.imshow("Matching", img_matches)
        cv2.waitKey(1)

    frame_count += 1

video.release()
cv2.destroyAllWindows()

print("Statystyki dopasowań:")
for filename, count in match_counts.items():
    print(f"{filename}: {count} dopasowań")

# zad 4

orb = cv2.ORB_create() # Initialize ORB Detector
key_images_orb = []

for file in sorted(glob.glob(os.path.join(key_images_path, "*.jpg"))):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    mask = np.ones_like(image, dtype=np.uint8) * 255 # Create a mask for the image

    kp, des = orb.detectAndCompute(image, mask) # Detect keypoints and compute descriptors using ORB
    key_images_orb.append((image, kp, des, file)) # Store image, keypoints, descriptors, and filename

video = cv2.VideoCapture("video.mp4")
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

match_counts_orb = {img[3]: 0 for img in key_images_orb}

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    best_match_img = None
    max_matches = 0
    best_matches = None

    for img, kp, des, filename in key_images_orb:
        if des is None or des_frame is None:
            continue
        matches = bf.match(des, des_frame)

        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > max_matches:
            max_matches = len(matches)
            best_match_img = img
            best_matches = matches
            best_filename = filename

    if best_match_img is not None:
        match_counts_orb[best_filename] += 1
        img_matches = cv2.drawMatches(best_match_img, key_images_orb[0][1], frame, kp_frame, best_matches[:10], None)
        cv2.imshow("Matching ORB", img_matches)
        cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()

print("Statystyki dopasowań (ORB):")
for filename, count in match_counts_orb.items():
    print(f"{filename}: {count} dopasowań")
