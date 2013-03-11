from os import path
import sys
import numpy as np
try:
    import cv2
except ImportError:
    print 'Couldn\'t find opencv so trying to use the fallback' \
          ' cv2.pyd (only for windows).'
    from _cv2_fallback import cv2


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Helper Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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
    return abs(area) / 2  # abs in case it's negative


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Fundamental Parts
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# alternative detectors, descriptors, matchers, parameters ==> different results
detector = cv2.BRISK(thresh=10, octaves=0)
extractor = cv2.DescriptorExtractor_create('BRISK')  # non-patented. Thank you!
matcher = cv2.BFMatcher(cv2.NORM_L2SQR)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Object Features
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
obj_original = cv2.imread(path.join('source_images', 'object.png'),
                          cv2.CV_LOAD_IMAGE_COLOR)
if obj_original is None:
    print 'Couldn\'t find the object image with the provided path.'
    sys.exit()


# basic feature detection works in grayscale
obj = cv2.cvtColor(obj_original, cv2.COLOR_BGR2GRAY)
# mask with white in areas to consider, black in areas to ignore
obj_mask = cv2.imread(path.join('source_images', 'object_mask.png'),
                      cv2.CV_LOAD_IMAGE_GRAYSCALE)
if obj_mask is None:
    print 'Couldn\'t find the object mask image with the provided path.' \
          ' Continuing without it.'


# keypoints are "interesting" points in an image:
obj_keypoints = detector.detect(obj, obj_mask)
# this lines up each keypoint with a mathematical description
obj_keypoints, obj_descriptors = extractor.compute(obj, obj_keypoints)
print 'Object Summary  *************************************************'
print '    {} keypoints'.format(len(obj_keypoints))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Scene Features
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
scene_original = cv2.imread(path.join('source_images', 'scene.png'),
                            cv2.CV_LOAD_IMAGE_COLOR)
if scene_original is None:
    print 'Couldn\'t find the scene image with the provided path.'
    sys.exit()


scene = cv2.cvtColor(scene_original, cv2.COLOR_BGR2GRAY)
scene_mask = cv2.imread(path.join('source_images', 'scene_mask.png'),
                        cv2.CV_LOAD_IMAGE_GRAYSCALE)
if scene_mask is None:
    print 'Couldn\'t find the scene mask image with the provided path.' \
          ' Continuing without it.'


scene_keypoints = detector.detect(scene, scene_mask)
scene_keypoints, scene_descriptors = extractor.compute(scene, scene_keypoints)
print 'Scene Summary  **************************************************'
print '    {} keypoints'.format(len(scene_keypoints))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Match features between the object and scene
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
min_matches = 3
matches = matcher.match(obj_descriptors, scene_descriptors)
if len(matches) < min_matches:
    print 'Not enough matches found between the image and scene keypoints.'
    sys.exit()


#do some filtering of the matches to find the best ones
distances = [match.distance for match in matches]
min_dist = min(distances)
avg_dist = sum(distances) / len(distances)
# basically allow everything except awful outliers
# a lower number like 2 will exclude a lot of matches if that's what you need
min_multiplier_tolerance = 10
min_dist = min_dist or avg_dist * 1.0 / min_multiplier_tolerance
good_matches = [match for match in matches if
                match.distance <= min_multiplier_tolerance * min_dist]
print 'Match Summary  **************************************************'
print '    {} / {}      good / total matches'.format(len(good_matches),
                                                     len(matches))
if len(good_matches) < min_matches:
    print 'not enough good matches to continue'
    sys.exit()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Calculate the shape of the object discovered in the scene.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#extract the positions of the good matches within the object and scene
obj_matched_points = np.array([obj_keypoints[match.queryIdx].pt
                               for match in good_matches])
scene_matched_points = np.array([scene_keypoints[match.trainIdx].pt
                                 for match in good_matches])
# find the homography which describes how the object is oriented in the scene
# also gets a mask which identifies each match as an inlier or outlier
homography, homography_mask = cv2.findHomography(obj_matched_points,
                                                 scene_matched_points,
                                                 cv2.RANSAC, 2.0)
print 'Homography Summary  **************************************************'
print'    {} / {}      inliers / good matches'.format(np.sum(homography_mask),
                                                      len(homography_mask))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Extract sizes and coordinates
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
obj_h, obj_w = obj.shape[0:2]
scene_h, scene_w = scene.shape[0:2]
#corners: opencv uses (top left, top right, bottom right, bottom left)
obj_top_left = (0, 0)
obj_top_right = (obj_w, 0)
obj_bottom_right = (obj_w, obj_h)
obj_bottom_left = (0, obj_h)
object_corners_float = np.array([obj_top_left, obj_top_right,
                                 obj_bottom_right, obj_bottom_left],
                                dtype=np.float32)
#corners of the object in the scene (I don't know about the reshaping)
obj_in_scene_corners_float =\
    cv2.perspectiveTransform(object_corners_float.reshape(1, -1, 2),
                             homography).reshape(-1, 2)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Visualize the matching results
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# create a combined image of the original object and the scene
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
combo_image = np.zeros((max(scene_h, obj_h), scene_w + obj_w), np.uint8)
combo_image[0:obj_h, 0:obj_w] = obj  # copy the obj into the combo image
combo_image[0:scene_h, obj_w:obj_w + scene_w] = scene  # same for the scene
combo_image = cv2.cvtColor(combo_image, cv2.COLOR_GRAY2BGR)  # color for output
# draw a polygon around the object in the scene
obj_in_scene_offset_corners_float = obj_in_scene_corners_float + (obj_w, 0)
cv2.polylines(combo_image, [np.int32(obj_in_scene_offset_corners_float)],
              True, blue, 2)
# mark inlier and outlier matches
for (x1, y1), (x2, y2), inlier in zip(np.int32(obj_matched_points),
                                      np.int32(scene_matched_points),
                                      homography_mask):
    if inlier:
        #draw a line with circle ends for each inlier
        cv2.line(combo_image, (x1, y1), (x2 + obj_w, y2), green)
        cv2.circle(combo_image, (x1, y1), 4, green, 2)
        cv2.circle(combo_image, (x2 + obj_w, y2), 4, green, 2)
    else:
        #draw a red x for outliers
        r = 2
        weight = 2
        cv2.line(combo_image,
                 (x1 - r, y1 - r), (x1 + r, y1 + r), red, weight)
        cv2.line(combo_image,
                 (x1 - r, y1 + r), (x1 + r, y1 - r), red, weight)
        cv2.line(combo_image,
                 (x2 + obj_w - r, y2 - r), (x2 + obj_w + r, y2 + r),
                 red, weight)
        cv2.line(combo_image,
                 (x2 + obj_w - r, y2 + r), (x2 + obj_w + r, y2 - r),
                 red, weight)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Do a sanity check on the discovered object
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
use_extracted = True  # keep track of whether the extraction works or not
#check for size of the discovered result versus the original
scale_tolerance = 0.7
obj_area = polygon_area(object_corners_float)
obj_in_scene_area = polygon_area(obj_in_scene_corners_float)
area_min_allowed = obj_area * (1 - scale_tolerance) ** 2
area_max_allowed = obj_area * (1 + scale_tolerance) ** 2
if not (area_min_allowed < obj_in_scene_area < area_max_allowed):
    print 'A homography was found but it seems too large or' \
          ' too small for a real match.'
    use_extracted = False


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Extract the object from the original scene
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#max/min the corners to disentangle any flipped, etc. projections
tops_bottoms = list(corner[1] for corner in obj_in_scene_corners_float)
lefts_rights = list(corner[0] for corner in obj_in_scene_corners_float)
#limit the boundaries to the scene boundaries
top = max(int(min(tops_bottoms)), 0)
bottom = min(int(max(tops_bottoms)), scene_h - 1)
left = max(int(min(lefts_rights)), 0)
right = min(int(max(lefts_rights)), scene_w - 1)
extracted = scene_original[top:bottom, left:right]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Display and save all the results
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
combo_path = path.join('.', 'output_images', 'match_visualization.png')
cv2.imwrite(combo_path, combo_image)
mini_combo_h, mini_combo_w = int(round(float(combo_image.shape[0])/2)),\
                             int(round(float(combo_image.shape[1])/2))
mini_combo = cv2.resize(combo_image, (mini_combo_w, mini_combo_h))
cv2.imshow('match visualization', mini_combo)
# only display/save extracted if the previous tests indicated it was realistic
if use_extracted:
    extracted_path = path.join('.', 'output_images', 'extracted.png')
    cv2.imwrite(extracted_path, extracted)
    mini_extr_h, mini_extr_w = int(round(float(extracted.shape[0])/2)),\
                               int(round(float(extracted.shape[1])/2))
    mini_extr = cv2.resize(extracted, (mini_extr_w, mini_extr_h))
    cv2.imshow('extracted image', mini_extr)


cv2.waitKey()
cv2.destroyAllWindows()
