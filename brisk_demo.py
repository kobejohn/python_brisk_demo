import sys
import numpy as np
import cv2


def showimage(image):
    """For playing around manually."""
    cv2.imshow('asdf', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def proportional_gaussian(image):
    """Help objects with differing sharpness look more similar to the feature detector etc."""
    kernel_w = int(2.0 * round((image.shape[1]*0.005+1)/2.0)-1)
    kernel_h = int(2.0 * round((image.shape[0]*0.005+1)/2.0)-1)
    return cv2.GaussianBlur(image, (kernel_w, kernel_h), 0) #blur to make features match at different resolutions


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Fundamental Parts
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#using alternative detectors, descriptors, matchers, and parameters will get different results
detector = cv2.FastFeatureDetector()
extractor = cv2.DescriptorExtractor_create('BRISK') #non-patented! Thank you!!!
matcher = cv2.BFMatcher(cv2.NORM_L2SQR)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Object Features
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
obj_original = cv2.imread('object.png', cv2.CV_LOAD_IMAGE_COLOR)
obj_gray = cv2.cvtColor(obj_original, cv2.COLOR_BGR2GRAY) #basic feature detection works in grayscale
obj = proportional_gaussian(obj_gray) #mild gaussian
#mask with white in areas to consider, black in areas to ignore
obj_mask = cv2.imread('object_mask.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
#this is the fingerprint:
obj_keypoints = detector.detect(obj, obj_mask)
obj_keypoints, obj_descriptors = extractor.compute(obj, obj_keypoints)
print 'Object Summary  *************************************************'
print '    {} keypoints'.format(len(obj_keypoints))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Scene Features
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
scene_original = cv2.imread('scene.png', cv2.CV_LOAD_IMAGE_COLOR)
scene_gray = cv2.cvtColor(scene_original, cv2.COLOR_BGR2GRAY)
scene = proportional_gaussian(scene_gray)
#you can use a mask like with the object if you want
#this is the fingerprint:
scene_keypoints = detector.detect(scene)
scene_keypoints, scene_descriptors = extractor.compute(scene, scene_keypoints)
print 'Scene Summary  **************************************************'
print '    {} keypoints'.format(len(scene_keypoints))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Match features between the object and scene
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
matches = matcher.match(obj_descriptors, scene_descriptors) #try to match similar keypoints
if len(matches) == 0:
    print 'No matches found between the image and scene keypoints.'
    sys.exit()
#do some filtering of the matches to find the best ones
distances = [match.distance for match in matches]
min_dist = min(distances)
avg_dist = sum(distances) / len(distances)
min_dist = min_dist or avg_dist * 0.01 #if min_dist is zero, use a small percentage of avg instead
good_matches = [match for match in matches if match.distance <= 3 * min_dist]
print 'Match Summary  **************************************************'
print '    {} / {}      good / total matches'.format(len(good_matches), len(matches))
if len(good_matches) < 3:
    print 'not enough good matches to continue'
    sys.exit()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Calculate the shape of the object discovered in the scene.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#extract the positions of the good matches within the object and scene
obj_matched_points = np.array([obj_keypoints[match.queryIdx].pt for match in good_matches])
scene_matched_points = np.array([scene_keypoints[match.trainIdx].pt for match in good_matches])
#find the homography which describes how the object is oriented in the scene
#this also gets the homography mask which identifies each match as an inlier or outlier
homography, homography_mask = cv2.findHomography(obj_matched_points, scene_matched_points, cv2.RANSAC, 2.0) #2.0: should be very close
print 'Homography Summary  **************************************************'
print'    {} / {}      inliers / total matches'.format(np.sum(homography_mask), len(homography_mask))
#it's basically done at this point. The rest is for visualization / making use of the results

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Visualize the results
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#next use the homography to identify the location of the object in the scene in pixel terms
#corners of the original object (opencv uses top left, top right, bottom right, bottom left)
obj_h, obj_w = obj.shape[0:2]
scene_h, scene_w = scene.shape[0:2]
object_corners = np.float32([(0,0), (obj_w, 0), (obj_w, obj_h), (0, obj_h)])
#corners of the object in the scene (I don't know about the reshaping. I never investigated it)
obj_in_scene_corners = cv2.perspectiveTransform(object_corners.reshape(1, -1, 2), homography).reshape(-1, 2)
#create a combined image of the original object and the scene
combo = np.zeros((max(scene_h, obj_h), scene_w + obj_w), np.uint8)
combo[0:obj_h, 0:obj_w] = obj
combo[0:scene_h, obj_w:obj_w+scene_w] = scene
combo = cv2.cvtColor(combo, cv2.COLOR_GRAY2BGR) #upgrade to color for output image
#draw a polygon around the object in the scene
obj_in_scene_offset_corners = np.int32(obj_in_scene_corners + (obj_w, 0)) #offset for the combined image
cv2.polylines(combo, [obj_in_scene_offset_corners], True, (0, 255, 0), 2)
#mark inlier and outlier matches
green = (0, 255, 0)
red = (0, 0, 255)
gray = (150, 150, 150)
for (x1, y1), (x2, y2), inlier in zip(np.int32(obj_matched_points), np.int32(scene_matched_points), homography_mask):
    if inlier:
        #draw a line with circle ends
        cv2.line(combo, (x1, y1), (x2+obj_w, y2), gray)
        cv2.circle(combo, (x1, y1), 4, green, 2)
        cv2.circle(combo, (x2+obj_w, y2), 4, green, 2)
    else:
        #draw a red x
        r = 2
        thickness = 2
        cv2.line(combo, (x1-r, y1-r), (x1+r, y1+r), red, thickness)
        cv2.line(combo, (x1-r, y1+r), (x1+r, y1-r), red, thickness)
        cv2.line(combo, (x2+obj_w-r, y2-r), (x2+obj_w+r, y2+r), red, thickness)
        cv2.line(combo, (x2+obj_w-r, y2+r), (x2+obj_w+r, y2-r), red, thickness)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Extract the object from the original scene
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
obj_in_scene_tl, obj_in_scene_tr, obj_in_scene_br, obj_in_scene_bl = obj_in_scene_corners
tops = (obj_in_scene_tl[1], obj_in_scene_tr[1])
bottoms = (obj_in_scene_bl[1], obj_in_scene_br[1])
lefts = (obj_in_scene_tl[0], obj_in_scene_bl[0])
rights = (obj_in_scene_tr[0], obj_in_scene_br[0])
#limit the boundaries to the scene boundaries
top =    max(min(tops), 0)
bottom = min(max(bottoms), obj_h - 1)
left =   max(min(lefts), 0)
right =  min(max(rights), obj_w - 1)
extracted = scene_original[top:bottom, left:right]
#display and save the matched and extracted images
cv2.imshow('match visualization', combo)
cv2.imwrite('match_visualization.png', combo)
cv2.imshow('extracted image', extracted)
cv2.imwrite('extracted.png', extracted)
cv2.waitKey()
cv2.destroyAllWindows()
