from os import path
import sys
import numpy as np
import cv2


def proportional_gaussian(image):
    """Help objects with differing sharpness look more similar to the feature detector etc."""
    kernel_w = int(2.0 * round((image.shape[1]*0.005+1)/2.0)-1)
    kernel_h = int(2.0 * round((image.shape[0]*0.005+1)/2.0)-1)
    return cv2.GaussianBlur(image, (kernel_w, kernel_h), 0) #blur to make features match at different resolutions

def polygon_area(vertices):
    """Calculate the area of the vertices described by the sequence of vertices.

    Thanks to Darel Rex Finley: http://alienryderflex.com/polygon_area/
    """
    area = 0.0
    X = [float(vertex[0]) for vertex in vertices]
    Y = [float(vertex[1]) for vertex in vertices]
    j = len(vertices) - 1
    for i in range(len(vertices)):
        area += (X[j] + X[i]) * (Y[j] - Y[i])
        j = i
    return abs(area) / 2 #abs in case it's negative


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
obj_original = cv2.imread(path.join('source_images', 'object.png'), cv2.CV_LOAD_IMAGE_COLOR)
if obj_original is None:
    print 'Couldn\'t find the object image with the provided path.'
    sys.exit()


obj_gray = cv2.cvtColor(obj_original, cv2.COLOR_BGR2GRAY) #basic feature detection works in grayscale
obj = proportional_gaussian(obj_gray) #mild gaussian
#mask with white in areas to consider, black in areas to ignore
obj_mask = cv2.imread(path.join('source_images', 'object_mask.png'), cv2.CV_LOAD_IMAGE_GRAYSCALE)
if obj_mask is None:
    print 'Couldn\'t find the object mask image with the provided path. Continuing without it.'


#this is the fingerprint:
obj_keypoints = detector.detect(obj, obj_mask)
obj_keypoints, obj_descriptors = extractor.compute(obj, obj_keypoints)
print 'Object Summary  *************************************************'
print '    {} keypoints'.format(len(obj_keypoints))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Scene Features
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
scene_original = cv2.imread(path.join('source_images', 'scene.png'), cv2.CV_LOAD_IMAGE_COLOR)
if scene_original is None:
    print 'Couldn\'t find the scene image with the provided path.'
    sys.exit()


scene_gray = cv2.cvtColor(scene_original, cv2.COLOR_BGR2GRAY)
scene = proportional_gaussian(scene_gray)
#mask with white in areas to consider, black in areas to ignore
scene_mask = cv2.imread('scene_mask.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
if scene_mask is None:
    print 'Couldn\'t find the scene mask image with the provided path. Continuing without it.'


#this is the fingerprint:
scene_keypoints = detector.detect(scene, scene_mask)
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Extract sizes and coordinates
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
obj_h, obj_w = obj.shape[0:2]
scene_h, scene_w = scene.shape[0:2]
#corners: opencv uses (top left, top right, bottom right, bottom left)
object_corners_float = np.float32([(0,0), (obj_w, 0), (obj_w, obj_h), (0, obj_h)])
#corners of the object in the scene (I don't know about the reshaping. I never investigated it)
obj_in_scene_corners_float = cv2.perspectiveTransform(object_corners_float.reshape(1, -1, 2), homography).reshape(-1, 2)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Visualize the matching results
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#create a combined image of the original object and the scene
combo = np.zeros((max(scene_h, obj_h), scene_w + obj_w), np.uint8)
combo[0:obj_h, 0:obj_w] = obj
combo[0:scene_h, obj_w:obj_w+scene_w] = scene
combo = cv2.cvtColor(combo, cv2.COLOR_GRAY2BGR) #upgrade to color for output image
#draw a polygon around the object in the scene
blue =  (255, 0, 0)
green = (0, 255, 0)
red =   (0, 0, 255)
gray =  (150, 150, 150)
obj_in_scene_offset_corners_float = obj_in_scene_corners_float + (obj_w, 0) #offset for the combined image
cv2.polylines(combo, [np.int32(obj_in_scene_offset_corners_float)], True, blue, 2)
#mark inlier and outlier matches
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
# Do a sanity check on the discovered object
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
use_extracted = True
#check for size of the discovered result versus the original
MAX_SCALE = 2
obj_area = polygon_area(object_corners_float)
obj_in_scene_area = polygon_area(obj_in_scene_corners_float)
if not (float(obj_area) / MAX_SCALE**2 < obj_in_scene_area < obj_area * MAX_SCALE**2):
    print 'A homography was found but it seems too large or small for a real match.'
    use_extracted = False

#check for crossings in the edges of the polygram made by the corners
#todo: this would be another good check

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Extract the object from the original scene
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#max/min the corners to disentangle any flipped, etc. projections
tops_bottoms = list(corner[1] for corner in obj_in_scene_corners_float)
lefts_rights = list(corner[0] for corner in obj_in_scene_corners_float)
#limit the boundaries to the scene boundaries
top =    max(int(min(tops_bottoms)), 0)
bottom = min(int(max(tops_bottoms)), scene_h - 1)
left =   max(int(min(lefts_rights)), 0)
right =  min(int(max(lefts_rights)), scene_w - 1)
extracted = scene_original[top:bottom, left:right]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Display and save all the results
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
cv2.imshow('match visualization', combo)
cv2.imwrite('match_visualization.png', combo)
if use_extracted:
    cv2.imshow('extracted image', extracted)
    cv2.imwrite('extracted.png', extracted)


cv2.waitKey()
cv2.destroyAllWindows()
