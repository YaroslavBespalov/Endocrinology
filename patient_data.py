import os
import shutil

import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm


class Patient():
    """Class for preprocessing patient data."""
    def __init__(self, path_to_patient_data):
        """Constructor."""
        self.path_to_patient_data = path_to_patient_data
        self.path_to_images = os.path.join(path_to_patient_data, 'Images')
        self.path_to_masks = os.path.join(path_to_patient_data, 'Masks')
        
    def get_patient_name(self):
        return os.path.basename(self.path_to_patient_data)
    
    def get_paths_to_images(self):
        return [os.path.join(self.path_to_images, image_name) for image_name in os.listdir(self.path_to_images)]
    
    def get_paths_to_masks(self):
        return [os.path.join(self.path_to_masks, mask_name) for mask_name in os.listdir(self.path_to_masks)]
    
    def read_tiff_file(self, path):
        image = Image.open(path)
        images = []
        frame_number = 0
        while True:
            try:
                image.seek(frame_number)
                images.append(np.array(image.convert("RGB")))
                frame_number += 1
            except EOFError:
                return np.array(images)
    
    def get_patient_data(self):
        patient_data = {}
        for path_to_image, path_to_mask in zip(tqdm(self.get_paths_to_images()), self.get_paths_to_masks()):
            name = os.path.splitext(os.path.basename(path_to_image))[0]
            patient_data[name] = {'image': self.read_tiff_file(path_to_image),
                                  'mask': self.read_tiff_file(path_to_mask)}
        
        return patient_data
    
    def preprocess_image(self, image):
        _, processed = cv2.threshold(image.copy(), 225, 255, cv2.THRESH_TOZERO_INV)

        return processed

    def find_largest_contour(self, image):
        processed = self.preprocess_image(image)
        contours, _ = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
    
        return contour

    def show_contour(self, image, contour, colour=(0, 255, 0)):
        img = image.copy()
        cv2.drawContours(img, [contour], 0, colour, 3)
        plt.imshow(img)
    
    def get_crop_mask(self, image):
        contour = self.find_largest_contour(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], (255, 255, 255))
    
        return mask

    def clean_image(self, image, mask):
        return cv2.bitwise_and(image, mask)
    
    def data_preprocessing(self, patient_data):
        for image_id in tqdm(patient_data.keys()):
            mask = self.get_crop_mask(patient_data[image_id]['image'][0])
            for frame_number in range(patient_data[image_id]['image'].shape[0]):
                patient_data[image_id]['image'][frame_number] = self.clean_image(patient_data[image_id]['image'][frame_number], mask)
    
    def save_tiff_images(self, patient_data, path_to_data):
        patient_name = self.get_patient_name()
        path_to_save = os.path.join(path_to_data, patient_name, 'Images')
        try: 
            os.makedirs(path_to_save)
        except OSError: 
            pass
        
        for image_id in tqdm(patient_data.keys()):
            images = [Image.fromarray(patient_data[image_id]['image'][i]) for i in range(patient_data[image_id]['image'].shape[0] - 1)]
            Image.fromarray(patient_data[image_id]['image'][0]).save(os.path.join(path_to_save, '{}.tif'.format(image_id)), save_all=True, append_images=images)
            
        path_to_save = os.path.join(path_to_data, patient_name, 'Masks')
        if os.path.exists(path_to_save):
            shutil.rmtree(path_to_save)
        shutil.copytree(self.path_to_masks, path_to_save)
    
    def make_dataset_table(self, patient_data, path_to_data):
        path_to_patient = os.path.join(path_to_data, self.get_patient_name())
        
        data = np.empty((0, 3))
        for image_id in patient_data.keys():
            for frame in range(patient_data[image_id]['image'].shape[0] - 1):
                path_to_image = os.path.join(path_to_patient, 'Images', '{}.tif'.format(image_id))
                path_to_mask = os.path.join(path_to_patient, 'Masks', '{}.labels.tif'.format(image_id))
                
                data = np.vstack((data, np.array([path_to_image, path_to_mask, frame])))
            
        return data
