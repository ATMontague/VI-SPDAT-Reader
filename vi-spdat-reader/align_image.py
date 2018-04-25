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
    # convert images to grayscale
    grey_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)