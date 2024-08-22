import matplotlib.pyplot as plt
import logging
import cv2
from genAlgo_idea_v2 import gen_Algo
import load_images
from registration_helper import highlight_pet_areas, prepare_overlay_image
from image_transformations import apply_transformations
import visualize_images


def main():
    # path to folder containing the images
    folder = '/Users/laurinerichlitzki/laurinesrepository/5SemGenAlgo/data'
    # load image pairs from the given folder
    image_pairs, patient_ids = load_images.load_images_from_folder(folder)
    # check if any image pairs were loaded
    if not image_pairs:
        logging.error("No image pairs found :(")
        return
    # process each pair of images
    for idx, (image1, image2) in enumerate(image_pairs):
        patient_id = patient_ids[idx]
        logging.info(f"Processing image pair {idx + 1}/{len(image_pairs)}")

        # apply genetic algorithm to find best transformation parameters to the second image
        best_params = gen_Algo(image1, image2)
        logging.info(f'Best transformation parameters for pair {idx + 1}: {best_params}')

        # apply best transformations to image2
        transformed_image = apply_transformations(image2, best_params)

        # resize transformed image to match image1
        transformed_image = cv2.resize(transformed_image, (image1.shape[1], image1.shape[0]))

        # highlight PET areas in the transformed image
        highlighted_pet_image = highlight_pet_areas(transformed_image)

        # prepare an overlay image by combining the image1 with the highlighted PET image
        overlay_image = prepare_overlay_image(image1, highlighted_pet_image)

        # visualize the original image and the overlay image
        visualize_images.visualize_images(image1, overlay_image, patient_id)


if __name__ == '__main__':
    main()
