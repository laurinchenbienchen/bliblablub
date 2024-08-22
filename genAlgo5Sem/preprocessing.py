import numpy as np
import cv2

# global variables for rectangle coordinates
rect_start = None
rect_end = None
drawing = False


# callback function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    global rect_start, rect_end, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        # start drawing rectangle
        rect_start = (x, y)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # update the end point of the rectangle as the mouse moves
            rect_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        # finish drawing the rectangle
        rect_end = (x, y)
        drawing = False


# function to allow the user to select a region on the image
def select_region(image):
    # allows the user to select a region on the image
    # :param image: the image on which the region is selected
    # :return: a list of region coordinates as tuples (x, y, width, height)
    regions = []

    def click_and_crop(event, x, y, flags, param):
        nonlocal startX, startY, cropping
        if event == cv2.EVENT_LBUTTONDOWN:
            # start cropping when the left mouse button is pressed
            startX, startY = x, y
            cropping = True
        elif event == cv2.EVENT_LBUTTONUP:
            # end cropping when the left mouse button is released
            endX, endY = x, y
            cropping = False
            # draw a rectangle around the selected region
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # append the region coordinates to the list
            regions.append((startX, startY, endX - startX, endY - startY))
            cv2.imshow("Image", image)

    startX, startY, cropping = -1, -1, False
    # display the image and set the mouse callback function
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_and_crop)

    # keep the window open until the user presses 'q'
    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    # close all opencv windows
    cv2.destroyAllWindows()
    return regions


# function to preprocess the image by keeping only specified regions
def preprocess_image(image, region_mask=None):
    # preprocesses the image by keeping only certain areas as defined by the region mask
    # :param image: input image (grayscale image)
    # :param region mask: optional
    # :return: preprocessed image
    mask = np.zeros(image.shape, dtype=np.uint8)

    if region_mask is not None:
        # create a mask where he specified regions are white
        for (x, y, w, h) in region_mask:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)
    # apply the mask to the image
    preprocessed_image = cv2.bitwise_and(image, image, mask=mask)
    return preprocessed_image
