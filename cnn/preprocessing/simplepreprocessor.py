# image preprocessor that resizes
# the image, ignoring the aspect ratio
# this preprocessor is by definition very basic – all we are doing is accepting an input
# image, resizing it to a fixed dimension, and then returning it.

# import the necessary packages
import cv2


class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect
        # ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
