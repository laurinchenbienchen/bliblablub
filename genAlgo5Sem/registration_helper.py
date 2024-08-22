import cv2


# Function to highlight PET image areas
def highlight_pet_areas(image):
    # highlight areas in pet image where pixel values exceed a threshold
    # defined threshold value to be processed
    threshold = 190
    # define the color to highlight
    blue_color = [0, 0, 255]
    # convert the grayscale image to bgr image (3 channels)
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # set pixels above the threshold to the color
    colored_image[image > threshold] = blue_color
    return colored_image


# overlay
def prepare_overlay_image(ct_image, highlighted_pet_image, alpha=128):
    # prepares an overlay image by combining a Ct image with a highlighted PET image
    # Convert the CT image to RGBA
    ct_image_rgba = cv2.cvtColor(ct_image, cv2.COLOR_GRAY2BGRA)

    # Convert the highlighted PET image to RGBA
    highlighted_pet_image_rgba = cv2.cvtColor(highlighted_pet_image, cv2.COLOR_BGR2BGRA)

    # Set alpha values for transparency
    highlighted_pet_image_rgba[:, :, 3] = alpha

    # combine the CT image and highlighted PET image using weighted addition
    # cv2.addWeighted function blends the two images based on their weights
    overlay_image = cv2.addWeighted(ct_image_rgba, 1.0, highlighted_pet_image_rgba, 0.5, 0)

    return overlay_image
