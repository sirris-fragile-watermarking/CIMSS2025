from custom_spatial_based import embed_watermark as embed_hilo
from custom_spatial_based import tamper_detection as tamper_detection_hilo
from custom_spatial_based import recovery as recovery_hilo

from grayscale_recovery import watermarking as embed_gray
from grayscale_recovery import tamper_detection as tamper_detection_gray
from grayscale_recovery import recovery as recovery_gray

import os
import imageio.v2 as imageio
from generic_attacks import cropping_attack
from help_functions import display_images

# Load test image
def load_and_crop_target():

    current_directory = os.path.dirname(__file__)
    img_path = os.path.join(current_directory, 'Images/visdrone.jpg')
    img = imageio.imread(img_path)

    edge_size = 256
    x_start = 80 + (600-256 - 150)
    y_start = 200
    crop = img[x_start:x_start + edge_size,y_start:y_start+edge_size]

    return crop

def store_results(img, name, extension):
    current_directory = os.path.dirname(__file__)
    img_path = os.path.join(current_directory, 'demo/{}.{}'.format(name, extension))
    imageio.imwrite(uri=img_path, im=img, format=extension)

if __name__ == "__main__":
    base = load_and_crop_target()

    # Do watermarking
    seed = 123

    tamper_x = 180 # 400
    tamper_y = 140 # 210

    w_hilo = embed_hilo(base, seed=seed)
    w_gray = embed_gray(base, seed=seed)
    
    attacked_hilo, _ = cropping_attack(w_hilo, x_start=tamper_x, y_start=tamper_y, x_len=25, y_len=65)
    attacked_gray, _ = cropping_attack(w_gray, x_start=tamper_x, y_start=tamper_y, x_len=25, y_len=65)

    # Tamper detection & recovery
    _, tamper_mask_hilo, auth_loc_hilo, data_loc_hilo, rng_hilo, received_data_array_hilo = tamper_detection_hilo(input_img=attacked_hilo, seed=seed)
    recovered_hilo = recovery_hilo(attacked_hilo, tamper_mask=tamper_mask_hilo, auth_loc=auth_loc_hilo, data_loc=data_loc_hilo, rng=rng_hilo, received_data_array=received_data_array_hilo)
    
    tamper_mask_gray, rec_indices_gray, recovery_watermark_gray = tamper_detection_gray(attacked_gray, seed=seed)
    recovered_gray = recovery_gray(attacked_gray, tamper_mask=tamper_mask_gray, recovery_indices=rec_indices_gray, recovery_watermark=recovery_watermark_gray)

    # Save images:
    store_results(base, name="start", extension="jpg")
    store_results(attacked_gray, name="attacked", extension="jpg")
    store_results(recovered_hilo, name="hilo", extension="jpg")
    store_results(recovered_gray, name="gray", extension="jpg")

    