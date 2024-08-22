import numpy as np
import cv2


# translation
def translate_image(image, tx, ty):
    # translates an image by specified amount in tx and ty direction
    # define the transformation matrix for translation
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    # apply translation using the cv2.warpAffine
    translated_image = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]))
    return translated_image


# rotation
def rotate_image(image, angle):
    # rotates an image by specified angle around its center
    # get the height and width of the image
    (h, w) = image.shape[:2]
    # define the center of rotation
    center = (w / 2, h / 2)
    # apply the rotation using cv2
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, m, (h, w))
    return rotated_image


# scaling
def scale_image(image, factor):
    # scales an image by a specific factor
    # resize the image using scaling factor for both dimensions (x,y)
    scaled_image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    return scaled_image
