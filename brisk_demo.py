from operator import attrgetter
import sys

import numpy as np
import cv2
import time

def showimage(image):
    cv2.namedWindow('asdf')
    cv2.imshow('asdf', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


#everything that can be done and stored ahead of time:
obj_original = cv2.imread('vs_obj_800x600.png', cv2.CV_LOAD_IMAGE_COLOR)
obj = cv2.cvtColor(obj_original, cv2.COLOR_BGR2GRAY)
obj_mask = cv2.imread('vs_obj_mask_800x600.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
#detector = cv2.FeatureDetector_create("FAST")
#detector = cv2.BRISK()
#detector = cv2.ORB()
detector = cv2.FastFeatureDetector(20)
obj_keypoints = detector.detect(obj, obj_mask)
print '************************************************************'
print 'scene - {} keypoints'.format(len(obj_keypoints))

extractor = cv2.DescriptorExtractor_create('BRISK')
obj_keypoints, obj_descriptors = extractor.compute(obj, obj_keypoints)
obj_h, obj_w = obj.shape[:2]
object_corners = np.float32([(0,0), (obj_w, 0), (obj_w, obj_h), (0, obj_h)]) #start corners are in shape of obj

#get the scene to be searched
scene_original = cv2.imread('scene_with_vs_640x480.png', cv2.CV_LOAD_IMAGE_COLOR)
scene = cv2.cvtColor(scene_original, cv2.COLOR_BGR2GRAY)

#detect keypoints
tb = time.time()
scene_keypoints = detector.detect(scene)
ta = time.time()
print '************************************************************'
print '{}s: get scene keypoints'.format(ta-tb)
print 'scene - {} keypoints'.format(len(scene_keypoints))

#compute descriptors
tb = time.time()
scene_keypoints, scene_descriptors = extractor.compute(scene, scene_keypoints)
ta = time.time()
print '************************************************************'
print '{}s: get scene descriptors'.format(ta-tb)

#match descriptors
matcher = cv2.BFMatcher(cv2.NORM_L1)
tb = time.time()
matches = matcher.match(obj_descriptors, scene_descriptors)
ta = time.time()
print '************************************************************'
print '{}s: match descriptors brute force'.format(ta-tb)

#identify 'good' matches
tb = time.time()
min_dist = min(matches, key=attrgetter('distance')).distance
max_dist = max(matches, key=attrgetter('distance')).distance
good_matches = [match for match in matches if match.distance < 3*min_dist]
ta = time.time()
print '************************************************************'
print '{}s: find good matches'.format(ta-tb)
print '{} good matches found'.format(len(good_matches))
if len(good_matches) < 3:
    print 'not enough good matches to continue'
    sys.exit()

#get positions of matches  and homography
tb = time.time()
obj_matched_points = np.array([obj_keypoints[match.queryIdx].pt for match in good_matches])
scene_matched_points = np.array([scene_keypoints[match.trainIdx].pt for match in good_matches])

#get the homography and homography_mask of the matched positions
homography, homography_mask = cv2.findHomography(obj_matched_points, scene_matched_points, cv2.RANSAC, 2.0) #2.0: should be very close
ta = time.time()
print '************************************************************'
print'{}s: get respective positions and homography'.format(ta-tb)
print'{} / {} inliers/matches'.format(np.sum(homography_mask), len(homography_mask))

#create a combined image for showing the results
scene_h, scene_w = scene.shape[:2]
vis = np.zeros((max(scene_h, obj_h), scene_w + obj_w), np.uint8)
vis[:obj_h, :obj_w] = obj
vis[:scene_h, obj_w:obj_w+scene_w] = scene
vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR) #upgrade to color for output image

#draw a box around the match to make it clear
scene_corners = cv2.perspectiveTransform(object_corners.reshape(1, -1, 2), homography).reshape(-1, 2) #reshaping removes extra list layers
scene_offset_corners = np.int32(scene_corners + (obj_w, 0)) #offset for the combined image
#print 'scene offset corners: {}'.format(scene_offset_corners)
cv2.polylines(vis, [scene_offset_corners], True, (0, 255, 0), 2)

#mark inclier and outlier matches on the combined image
if homography_mask is None:
    homography_mask = np.ones(len(scene_matched_points), np.bool)


green = (0, 255, 0)
red = (0, 0, 255)
gray = (150, 150, 150)
for (x1, y1), (x2, y2), inlier in zip(np.int32(obj_matched_points), np.int32(scene_matched_points), homography_mask):
    if inlier:
        #draw a line with circle ends
        cv2.line(vis, (x1, y1), (x2+obj_w, y2), gray)
        cv2.circle(vis, (x1, y1), 4, green, 2)
        cv2.circle(vis, (x2+obj_w, y2), 4, green, 2)
    else:
        #draw a red x
        r = 2
        thickness = 2
        cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), red, thickness)
        cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), red, thickness)
        cv2.line(vis, (x2+obj_w-r, y2-r), (x2+obj_w+r, y2+r), red, thickness)
        cv2.line(vis, (x2+obj_w-r, y2+r), (x2+obj_w+r, y2-r), red, thickness)


#extract a maximum size image from the probably non-rectangular bounding corners
scene_top_left, scene_top_right, scene_bottom_right, scene_bottom_left = scene_offset_corners
tops = (scene_top_left[1], scene_top_right[1])
bottoms = (scene_bottom_left[1], scene_bottom_right[1])
lefts = (scene_top_left[0], scene_bottom_left[0])
rights = (scene_top_right[0], scene_bottom_right[0])
top = min(tops)
bottom = max(bottoms)
left = min(lefts) - obj_w
right = max(rights) - obj_w
extracted = scene_original[top:bottom, left:right]

#display and save the matched and extracted images
cv2.imshow('match visualization', vis)
cv2.imwrite('match_visualization.png', vis)
cv2.imshow('extracted image', extracted)
cv2.imwrite('extracted.png', extracted)
cv2.waitKey()
cv2.destroyAllWindows()
