import cv2
import numpy as np


def ALIKED_SIM(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def LightGlue(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return matches

def filter_confidence(matches, threshold=0.8):
    distances = [m.distance for m in matches]
    limit = np.percentile(distances, threshold * 100)
    return [m for m in matches if m.distance < limit]


img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Feature extraction
kp1, desc1 = ALIKED_SIM(img1)
kp2, desc2 = ALIKED_SIM(img2)

# 2. Matching
matches = LightGlue(desc1, desc2)

# 3. Filter matches by confidence (simulated)
filtered = filter_confidence(matches, threshold=0.8)

# 4. Visualize matches
match_img = cv2.drawMatches(img1, kp1, img2, kp2, filtered, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("ALIKED_SIM + LightGlue (Simulated)", match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
