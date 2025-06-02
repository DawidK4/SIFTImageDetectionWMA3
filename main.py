import cv2
import os

# Ścieżki i tworzenie folderów
photos_path = 'photos'
video_path = 'video.mp4'

folders = {
    'sift_masks': 'masks_sift',
    'orb_masks': 'masks_orb',
    'sift_output': 'output_sift',
    'orb_output': 'output_orb'
}
for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

# --- Funkcja do tworzenia maski (używana dla obu metod) ---
def create_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = (0, 20, 50)
    upper = (180, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# --- Funkcja ekstrakcji cech (SIFT lub ORB) ---
def extract_features(detector):
    """
    Extracts keypoints and descriptors from reference images using the given detector.

    Args:
        detector: A feature detector object (e.g., SIFT or ORB) with a detectAndCompute method.

    Returns:
        tuple: Three lists:
            - imgs: Loaded images (BGR format).
            - kps: Detected keypoints for each image.
            - dess: Descriptors for each image.

    The function processes images named 'key1.jpg' to 'key4.jpg' in the 'photos' directory.
    For each image:
        - Loads the image.
        - Converts it to grayscale.
        - Detects keypoints and computes descriptors.
        - Appends the results to the respective lists.
    If an image cannot be loaded, it prints an error and skips it.
    """
    imgs = []
    kps = []
    dess = []

    for i in range(1, 5):
        img_path = os.path.join(photos_path, f'key{i}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Nie można wczytać: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = detector.detectAndCompute(gray, None)
        imgs.append(img)
        kps.append(kp)
        dess.append(des)
    return imgs, kps, dess

# --- Funkcja zapisu masek ---
def save_masks(mask_folder):
    for i in range(1, 5):
        img_path = os.path.join(photos_path, f'key{i}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            continue
        mask = create_mask(img)
        cv2.imwrite(os.path.join(mask_folder, f'mask{i}.png'), mask)

# --- Funkcja dopasowania i zapisu wyników ---
def match_and_save(detector_name, detector, norm_type, mask_folder, output_folder):
    print(f"\n--- Przetwarzanie metodą {detector_name} ---")

    key_imgs, key_kps, key_dess = extract_features(detector)
    save_masks(mask_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Nie można otworzyć pliku wideo: {video_path}")
        return

    bf = cv2.BFMatcher(norm_type, crossCheck=(detector_name=='ORB'))

    frame_count = 0
    match_stats = [0] * len(key_dess)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_des = detector.detectAndCompute(gray_frame, None)

        best_idx = -1
        max_matches = 0
        best_matches = []

        for idx, des in enumerate(key_dess):
            if des is None or frame_des is None:
                continue
            if detector_name == 'SIFT':
                matches = bf.knnMatch(des, frame_des, k=2)
                # Lowe's ratio test
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            else:  # ORB
                matches = bf.match(des, frame_des)
                good = sorted(matches, key=lambda x: x.distance)

            if len(good) > max_matches:
                best_idx = idx
                max_matches = len(good)
                best_matches = good

        if best_idx != -1:
            match_stats[best_idx] += 1
            # Rysowanie dopasowań (maks 10)
            result = cv2.drawMatches(
                key_imgs[best_idx],
                key_kps[best_idx],
                frame,
                frame_kp,
                best_matches[:10],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            out_frame_path = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(out_frame_path, result)

        frame_count += 1

    cap.release()

    print(f"\nStatystyki dopasowania dla {detector_name}:")
    for i, count in enumerate(match_stats):
        print(f" key{i+1}.jpg: {count} klatek")

# --- Uruchomienie dla SIFT i ORB ---

# SIFT
sift_detector = cv2.SIFT_create()
match_and_save(
    detector_name='SIFT',
    detector=sift_detector,
    norm_type=cv2.NORM_L2,
    mask_folder=folders['sift_masks'],
    output_folder=folders['sift_output']
)

# ORB
orb_detector = cv2.ORB_create(nfeatures=1000)
match_and_save(
    detector_name='ORB',
    detector=orb_detector,
    norm_type=cv2.NORM_HAMMING,
    mask_folder=folders['orb_masks'],
    output_folder=folders['orb_output']
)
