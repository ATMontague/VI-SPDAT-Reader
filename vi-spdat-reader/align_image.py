"""
Created on Tue Apr 24 18:54:06 2018

@author: atmon
"""
# from tutorial
# https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
from __future__ import print_function
import cv2
import numpy as np
from wand.image import Image

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def align_image(image1, image2):
    """Takes two images, returns homography and corrected image"""
    # convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # detect ORB features and compute descriptors
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)
    
    # match features
    matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
    matches = matcher.match(descriptors1, descriptors2, None)

    # sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # remove poor matches
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]
    
    # draw top matches
    image_matches = cv2.drawMatches(image1, keypoints1, image2,
                                keypoints2, matches, None)
    cv2.imwrite('matches.jpg', image_matches)
    
    # extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        
    # find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # use homography
    height, width, channels = image2.shape
    registered_image = cv2.warpPerspective(image1, h, (width, height))
    
    return registered_image, h
    
def convert_to_image(fname):
    """Receives pdf and returns png"""
    
    # only handling pdf for now
    if not fname.lower().endswith('.pdf'):
        return None
    with Image(filename=fname, resolution=300) as img:
        # converting to png
        # looks into weird pdf error
        img.format = 'png'
        img.alpha_channel = 'remove'
        img.save(filename='test.png')
    
    
if __name__ == '__main__':
    
    # read reference image
    reference_filename = 'page1.jpg'
    reference_image = cv2.imread(reference_filename, cv2.IMREAD_COLOR)
    
    # read image to be aligned
    imperfect_filename = 'tilted_image1.jpg'
    imperfect_image = cv2.imread(imperfect_filename, cv2.IMREAD_COLOR)
    
    # get aligned image and homography
    aligned_image, h = align_image(imperfect_image, reference_image)
    
    # write aligned image to disk
    outfile = 'aligned.jpg'
    cv2.imwrite(outfile, aligned_image)
    
    # print estimated homography
    print('Estimated homography: {}'.format(h))
    
    # testing converting pdf to png
    print("Converting to png")
    pdf = 'page1.pdf'
    convert_to_image(pdf)
    print('Done')
