import transformations
import cv2


# function to apply transformations to an image
def apply_transformations(image, params):
    # apply translation using the parameter 'tx' und 'ty'
    transformed_image = transformations.translate_image(image, params['tx'], params['ty'])
    # apply rotation to the translated image using the parameter 'angle'
    transformed_image = transformations.rotate_image(transformed_image, params['angle'])
    # apply scaling to the rotated and translated image using the parameter 'scale'
    transformed_image = transformations.scale_image(transformed_image, params['scale'])

    # resize transformed image to match the reference image
    return cv2.resize(transformed_image, (image.shape[1], image.shape[0]))
