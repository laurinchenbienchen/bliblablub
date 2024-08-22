import os
import cv2
from region_selector import RegionSelector
from preprocessing import preprocess_image


def load_images_from_folder(main_folder):
    # list to store pairs of images
    image_pairs = []
    patient_ids = []
    # iterate through each subfolder in the main folder
    for patient_folder in os.listdir(main_folder):
        patient_path = os.path.join(main_folder, patient_folder)
        # check if the path is a directory
        if os.path.isdir(patient_path):
            print(f"Processing folder: {patient_path}")
            # construct file paths for CT and PET images
            ct_img_path = os.path.join(patient_path, f'{patient_folder}_ct.png')
            pet_img_path = os.path.join(patient_path, f'{patient_folder}_pt.png')
            # check if both image files exist
            if os.path.exists(ct_img_path) and os.path.exists(pet_img_path):
                # read the images in grayscale mode
                ct_img = cv2.imread(ct_img_path, cv2.IMREAD_GRAYSCALE)
                pet_img = cv2.imread(pet_img_path, cv2.IMREAD_GRAYSCALE)
                # check if images were successfully read
                if ct_img is not None and pet_img is not None:
                    patient_id = patient_folder.split('p')[1]
                    patient_ids.append(patient_id)
                    # user selects a region for the CT images
                    print("Select region for CT image")
                    # create RegionSelector object for CT images
                    selector_ct = RegionSelector(ct_img)
                    # user selects region
                    region_mask_ct = selector_ct.select_region()
                    # debugg output
                    print(f"Region mask for CT image: {region_mask_ct}")
                    if region_mask_ct:
                        # preprocess CT image
                        ct_img = preprocess_image(ct_img, region_mask=region_mask_ct)
                    else:
                        print("No region selected for CT image. Skipping...")
                        continue

                    # user selects region for PET image
                    print("Select region for PET image")
                    # create RegionSelector object for PET image
                    selector_pet = RegionSelector(pet_img)
                    # user selects region
                    region_mask_pet = selector_pet.select_region()
                    # debug output
                    print(f"Region mask for PET image: {region_mask_pet}")
                    if region_mask_pet:
                        # preprocess PET image
                        pet_img = preprocess_image(pet_img, region_mask=region_mask_pet)
                    else:
                        print("No region selected for PET image. Skipping...")
                        continue
                    # append the preprocessed CT and PET images as a pair to the list
                    image_pairs.append((ct_img, pet_img))
                else:
                    print(f"Failed to read images in {patient_folder}")
            else:
                print(f"Image files not found in {patient_folder}")
    return image_pairs, patient_ids
