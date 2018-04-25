# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 18:54:06 2018

@author: atmon
"""

from __future__ import print_function
import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def align_images(image1, image2):
    """Takes two images and returns homography and corrected image"""
    # convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # detect ORB features and compute descriptors
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)
    
    # match features
    matcher = cv2.DescriptorMatcher_create
    (cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # remove poor matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    
    # draw top matches
    imMatches = cv2.drawMatches(image1, keypoints1, image2,
                                keypoints2, matches, None)
    cv2.imwrite('matches.jpg', imMatches)
    
    # extract location of good matches
    points1 = np.zeroes((len(matches), 2), dtype=np.float32)
    points2 = np.zeroes((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        
    # find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # use homography
    height, width, channels = image2.shape
    im1Reg = cv2.warpPerspective(image1, h, (width, height))
    
    return im1Reg, h

    