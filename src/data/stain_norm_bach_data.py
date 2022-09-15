'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''


from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization
from histomicstk.preprocessing.color_deconvolution.\
    color_deconvolution import color_deconvolution_routine, stain_unmixing_routine
from histomicstk.preprocessing.augmentation.\
    color_augmentation import rgb_perturb_stain_concentration, perturb_stain_concentration

from PIL import Image
import os, cv2
import numpy as np
    
# color norm. standard (from TCGA-A2-A3XS-DX1, Amgad et al, 2019)
cnorm = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}

images_path = "/home/datasets/BACH/images"
normalized_images_path = "/home/datasets/BACH/images_stain_normalized"

classes_list = ["benign","insitu","invasive","normal"]

for category in classes_list:
    for image in os.listdir(os.path.join(images_path, category)):
        print("normalization for -", os.path.join(images_path, category, image))
        data_image = np.asarray(Image.open(os.path.join(images_path, category, image)))
        tissue_rgb_normalized = reinhard(data_image, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'])
        cv2.imwrite(os.path.join(normalized_images_path, category, image), tissue_rgb_normalized)
  
