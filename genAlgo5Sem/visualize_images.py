import matplotlib.pyplot as plt
import cv2


def visualize_images(image1, overlay_image, patient_id):
    # visualizes two images side by side using matplotlib
    # create a new figure with a specified size
    plt.figure(figsize=(10, 5))
    # plot the first image (CT image) on the left
    plt.subplot(1, 2, 1)
    plt.title(f"CT Image - Patient {patient_id}")
    plt.imshow(image1, cmap='gray')
    # plot the second image (overlay image) on the right
    plt.subplot(1, 2, 2)
    plt.title("Overlay Image with Highlighted PET Areas - Patient {patient_id}")
    # convert the overlay image from RGBA to RGB for correct display
    plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_RGBA2RGB))
    # show the plot
    plt.show()
