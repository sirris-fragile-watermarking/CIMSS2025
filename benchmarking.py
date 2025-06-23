"""
Own code that evaluates existing and new fragile watermarking schemes
"""
import secrets
import os
import shutil
import json
import imageio.v2 as imageio
from enum import Enum
from generic_attacks import *
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio

from help_functions import display_images, BinaryClassification
import numpy as np
from time import time

from median_method import do_self_embedding_rgb as embed_median
from median_method import tamper_detection as tamper_detection_median
from median_method import recovery as recovery_median

from original_code_interface import do_fast_embedding as embed_double_dwt_fast
from original_code_interface import do_fast_tamper_detection as tamper_detection_double_dwt_fast
from original_code_interface import do_fast_recovery as recovery_double_dwt_fast

from custom_spatial_based import embed_watermark as embed_hi_lo_spatial
from custom_spatial_based import tamper_detection as tamper_detection_hi_lo_spatial
from custom_spatial_based import recovery as recovery_hi_lo_spatial

from svd_based import do_svd_and_block_based_watermarking as embed_svd
from svd_based import tamper_detection as tamper_detection_svd
from svd_based import recovery as recovery_svd

from molina_garcia import watermark_embedding as embed_molina_garcia
from molina_garcia import tamper_detection as tamper_detection_molina_garcia
from molina_garcia import recovery as recovery_molina_garcia

from lbp_based import do_self_embedding_rgb as embed_sisaudia
from lbp_based import tamper_detection as tamper_detection_sisaudia
from lbp_based import recovery as recovery_sisaudia

from grayscale_recovery import watermarking as embed_grayscale
from grayscale_recovery import tamper_detection as tamper_detection_grayscale
from grayscale_recovery import recovery as recovery_grayscale

BENCHMARKS_PATH = os.path.join(os.path.dirname(__file__), "benchmarking")
BENCHMARK_IMAGES_PATH = os.path.join(BENCHMARKS_PATH, "evaluation_image_results")

def store_image(image, path, name, format):
    uri = os.path.join(path, name + ".{}".format(format))
    imageio.imwrite(uri=uri, im=image, format=format)


def store_image(image, path, name, format):
    uri = os.path.join(path, name + ".{}".format(format))
    imageio.imwrite(uri=uri, im=image, format=format)

def create_or_append_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def create_clean_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def technique_to_filepath(technique):

    filename = technique + ".json"
    file_path = os.path.join(BENCHMARKS_PATH, filename)
    
    return file_path

def initialize_performance_metrics_file(technique):
    file_path = technique_to_filepath(technique=technique)
    
    if not os.path.isfile(file_path):
        with open(file_path, "w") as fp:
            start_dict = {"method": technique, "images": {}}
            json.dump(start_dict, fp)

def add_img_to_performance_metrics_file(technique, image_name, img_size_x, img_size_y, img_channels, format=None):
    file_path = technique_to_filepath(technique=technique)

    with open(file_path, "r") as fp:
        file_dict = json.load(fp)
    
    metadata_dict = {
        "size_x": img_size_x,
        "size_y": img_size_y,
        "channels": img_channels,
        "format": "" if format is None else format
    }

    if image_name not in file_dict["images"]:
        file_dict["images"][image_name] = {"metadata": metadata_dict, "benchmarks": {}}
    else:
        # Update metadata:
        file_dict["images"][image_name]["metadata"] = metadata_dict
    
    # Store dict again
    with open(file_path, "w+") as fp:
        json.dump(file_dict, fp)

def update_performance_metrics_file(technique, image_name, test_name, mse=-1, psnr=0, ssim=0, precision=-1, recall=-1, time=-1):
    file_path = technique_to_filepath(technique=technique)

    with open(file_path, "r") as fp:
        file_dict = json.load(fp)
    
    # Update dict
    # print(file_dict)
    mse = float(np.round(mse, decimals=4))
    psnr = float(np.round(psnr, decimals=4))
    ssim = float(np.round(ssim, decimals=4))
    time = float(np.round(time, decimals=4))
    precision = float(np.round(precision, decimals=4))
    recall = float(np.round(recall, decimals=4))

    file_dict["images"][image_name]["benchmarks"][test_name] = {"precision": precision, "recall": recall, 
                                                                "MSE": mse, "PSNR": psnr, "SSIM": ssim, "time": time}
    # print(file_dict)

    # Store dict again
    with open(file_path, "w+") as fp:
        json.dump(file_dict, fp)

def benchmark_method_time(input_image, method, kwargs, n_iterations=100):
    t1 = time()
    for _ in range(n_iterations):
        method(input_image, **kwargs)
    
    t2 = time()
    avg_time = np.round((t2 - t1)/n_iterations, decimals=4)

    return avg_time

def tamper_detection_evaluation(true_tamper_mask, pred_tamper_mask):
    # Turn masks into lists
    true_list = np.reshape(true_tamper_mask, -1)
    pred_list = np.reshape(pred_tamper_mask, -1)

    return BinaryClassification(y_true=true_list, y_pred=pred_list)

def image_metrics(image_true, image_test):

    mse = mean_squared_error(image0=image_true, image1=image_test)

    psnr = peak_signal_noise_ratio(image_true=image_true, image_test=image_test)
    
    # For non-square images: The SSIM should use a window size with odd values smaller than the smallest side of the image
    # For now, use 11x11 windows
    ssim = structural_similarity(im1=image_true, im2=image_test, win_size=11, data_range=255, channel_axis=2)

    return mse, psnr, ssim

def process_attack(original, watermarked, attacked, detection_method, recovery_method,
                   attack_name, attack_mask, technique, image_name,  storage_dir, 
                   benchmark_recovery_time=0, display=False):

    detected_mask, kwargs = detection_method(attacked)
    recovered = recovery_method(attacked, **kwargs)

    avg_time = -1

    # If indicated, benchmark the average recovery time
    if benchmark_recovery_time > 0:
        t1 = time()

        for _ in range(benchmark_recovery_time):
            recovered = recovery_method(attacked, **kwargs)
        
        t2 = time()
        avg_time = np.round(t2 - t1, decimals=4)

    # create_clean_dir()
    store_image(image=attacked, path=storage_dir, name="attack_{}".format(attack_name), format="png")
    store_image(image=recovered, path=storage_dir, name="recover_{}".format(attack_name), format="png")

    # Compute tamper detection metrics
    tamper_eval = tamper_detection_evaluation(true_tamper_mask=attack_mask, pred_tamper_mask=detected_mask)

    # Compute recovery metrics
    if original is not None:
        rec_mse, rec_psnr, rec_ssim = image_metrics(image_true=original, image_test=recovered)
    else:    
        rec_mse, rec_psnr, rec_ssim = image_metrics(image_true=watermarked, image_test=recovered)

    if display:
        text = "Detection performance: Precision: {}, Recall: {}\n".format(
            np.round(tamper_eval.precision(), 4), 
            np.round(tamper_eval.recall(), 4))
        text += "Recovery from crop attack: MSE: {}, PSNR: {}, SSIM: {}".format(np.round(rec_mse, 2), np.round(rec_psnr, 2), np.round(rec_ssim, 2))
        print(text)
        display_images([watermarked, attacked, recovered], text)

    update_performance_metrics_file(technique=technique, image_name=image_name, test_name=attack_name,
                                    mse=rec_mse, psnr=rec_psnr, ssim=rec_ssim, time=avg_time,
                                    precision=tamper_eval.precision(), recall=tamper_eval.recall())

    return rec_mse, rec_psnr, rec_ssim

def test_random_crop_center_attack(original, watermarked, detection_method, recovery_method, 
                                   technique, image_name,  storage_dir, percentage=0.5, 
                                   benchmark_recovery_time=0, display=False):
    base_x, base_y = original.shape[:2]

    crop_x = int(np.sqrt(percentage) * base_x)
    crop_y = int(np.sqrt(percentage) * base_y)
    start_x = (base_x - crop_x) // 2
    start_y = (base_y - crop_y) // 2
    attack_name = "center_crop_random_{}%".format(float(np.round(percentage * 100, decimals=2)))

    rng = np.random.default_rng(seed=secrets.randbits(128))
    random_values = rng.integers(low=0, high=256, size=(crop_x, crop_y, 3))

    attacked, attack_mask = cropping_attack(watermarked, x_start=start_x, y_start=start_y, 
                                            x_len=crop_x, y_len=crop_y, crop_value=random_values)
    
    return process_attack(original=original, watermarked=watermarked, attacked=attacked,
                          detection_method=detection_method, recovery_method=recovery_method,
                          attack_name=attack_name, attack_mask=attack_mask, technique=technique,
                          image_name=image_name, storage_dir=storage_dir,
                          benchmark_recovery_time=benchmark_recovery_time, display=display)

def test_value_crop_center_attack(original, watermarked, detection_method, recovery_method, 
                                  technique, image_name,  storage_dir, percentage=0.5, value=0,
                                  benchmark_recovery_time=0, display=False):
    base_x, base_y = original.shape[:2]

    crop_x = int(np.sqrt(percentage) * base_x)
    crop_y = int(np.sqrt(percentage) * base_y)
    start_x = (base_x - crop_x) // 2
    start_y = (base_y - crop_y) // 2
    
    # Clean the value
    value = int(np.clip(value, a_min=0, a_max=255))

    attack_name = "center_crop_{}_{}%".format(value, float(np.round(percentage * 100, decimals=2)))

    attacked, attack_mask = cropping_attack(watermarked, x_start=start_x, y_start=start_y, 
                                            x_len=crop_x, y_len=crop_y, crop_value=value)
    
    return process_attack(original=original, watermarked=watermarked, attacked=attacked,
                          detection_method=detection_method, recovery_method=recovery_method,
                          attack_name=attack_name, attack_mask=attack_mask, technique=technique,
                          image_name=image_name, storage_dir=storage_dir,
                          benchmark_recovery_time=benchmark_recovery_time, display=display)

def run_evaluation_suite_full(input_image, embedding_method, tamper_detection_method, recovery_method, 
                              n_timing_iterations, technique_name, image_name, format, do_benchmarking=True, do_attack_suite=True):
    # Prelims
    img_storage_path = os.path.join(BENCHMARK_IMAGES_PATH, technique_name)

    size_x, size_y = input_image.shape[:2]
    n_channels = 1 if (len(input_image.shape) == 2) else input_image.shape[-1]

    create_or_append_dir(img_storage_path)

    initialize_performance_metrics_file(technique=technique_name)
    
    add_img_to_performance_metrics_file(technique=technique_name, 
                                        image_name=image_name,
                                        img_size_x=size_x,
                                        img_size_y=size_y,
                                        img_channels=n_channels,
                                        format=format)
    # Start benchmarking measurements
    watermarked = embedding_method(input_image)
    base_mse, base_psnr, base_ssim = image_metrics(image_true=input_image, image_test=watermarked)

    if do_benchmarking:
        emb_time = benchmark_method_time(input_image=input_image, method=embedding_method, 
                                        kwargs={}, n_iterations=n_timing_iterations)
        
        update_performance_metrics_file(technique=technique_name,
                                        image_name=image_name,
                                        test_name="embedding",
                                        mse=base_mse,
                                        psnr=base_psnr,
                                        ssim=base_ssim,
                                        time=emb_time)

        det_time = benchmark_method_time(input_image=input_image, method=tamper_detection_method, 
                                        kwargs={}, n_iterations=n_timing_iterations)
        
        update_performance_metrics_file(technique=technique_name,
                                        image_name=image_name,
                                        test_name="tamper_detection",
                                        mse=-1,
                                        psnr=-1,
                                        ssim=-1,
                                        time=det_time)
    
    if do_attack_suite:
        crop_percentages = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        for percentage in crop_percentages:
            test_random_crop_center_attack(
                original=input_image, watermarked=watermarked, detection_method=tamper_detection_method, 
                recovery_method=recovery_method, 
                technique=technique_name, image_name=image_name, display=False, 
                benchmark_recovery_time=0, storage_dir=img_storage_path, percentage=percentage)
            
            test_value_crop_center_attack(
                original=input_image, watermarked=watermarked, detection_method=tamper_detection_method, 
                recovery_method=recovery_method, 
                technique=technique_name, image_name=image_name, display=False, value=0,
                benchmark_recovery_time=0, storage_dir=img_storage_path, percentage=percentage)


def evaluate_double_dwt_fast(input_image, image_name, format=None, do_benchmarking=True, do_attack_suite=True):
    # Housekeeping
    technique_name = "double_dwt_fast"
    
    def embedding_method(img):
        return embed_double_dwt_fast(input_image=img)
    
    def tamper_detection_method(img):
        tamper_fs, tamper, Rec_watermark1, Rec_watermark2, kwargs = tamper_detection_double_dwt_fast(input_image=img)        
        kwargs = {"tamper": tamper, "r1": Rec_watermark1, "r2": Rec_watermark2, "kwargs_dict": kwargs}

        return tamper_fs, kwargs

    def recovery_method(img, tamper, r1, r2, kwargs_dict):
        recovered = recovery_double_dwt_fast(input_image=img, tamper=tamper, r1=r1, r2=r2, kwargs=kwargs_dict)
        return recovered
    
    run_evaluation_suite_full(input_image=input_image, 
                              embedding_method=embedding_method,
                              tamper_detection_method=tamper_detection_method,
                              recovery_method=recovery_method, n_timing_iterations=100,
                              technique_name=technique_name, image_name=image_name, format=format,
                              do_benchmarking=do_benchmarking, do_attack_suite=do_attack_suite)


def evaluate_hi_lo_spatial(input_image, image_name, format=None, do_benchmarking=True, do_attack_suite=True):
    # Housekeeping
    technique_name = "hi_lo_spatial"
    seed = secrets.randbits(128)
    
    def embedding_method(img):
        return embed_hi_lo_spatial(input_img=img, seed=seed)
    
    def tamper_detection_method(img):
        tamper_mask_fs, tamper_mask, auth_loc, data_loc, rng, received_data_array = tamper_detection_hi_lo_spatial(input_img=img, seed=seed)
        kwargs = {"tamper_mask": tamper_mask, "auth_loc": auth_loc, "data_loc": 
                  data_loc, "rng": rng, "received_data_array": received_data_array}

        return tamper_mask_fs, kwargs

    def recovery_method(img, tamper_mask, auth_loc, data_loc, rng, received_data_array):
        recovered = recovery_hi_lo_spatial(input_img=img, tamper_mask=tamper_mask,
                                           auth_loc=auth_loc, data_loc=data_loc,
                                           rng=rng, received_data_array=received_data_array,
                                           sharpen=True)
        return recovered
    
    run_evaluation_suite_full(input_image=input_image, 
                              embedding_method=embedding_method,
                              tamper_detection_method=tamper_detection_method,
                              recovery_method=recovery_method, n_timing_iterations=100,
                              technique_name=technique_name, image_name=image_name, format=format,
                              do_benchmarking=do_benchmarking, do_attack_suite=do_attack_suite)

def evaluate_grayscale(input_image, image_name, format=None, do_benchmarking=True, do_attack_suite=True):
    # Housekeeping
    technique_name = "grayscale"
    seed = secrets.randbits(128)

    def embedding_method(img):
        return embed_grayscale(img=img, seed=seed)
    
    def tamper_detection_method(img):
        tamper_mask, rec_indices, recovery_watermark = tamper_detection_grayscale(img=img, seed=seed)
        kwargs = {"tamper_mask": tamper_mask, "recovery_indices": rec_indices, "recovery_watermark": recovery_watermark}

        return tamper_mask, kwargs

    def recovery_method(img, tamper_mask, recovery_indices, recovery_watermark):
        return recovery_grayscale(img=img, tamper_mask=tamper_mask, 
                                  recovery_indices=recovery_indices,
                                  recovery_watermark=recovery_watermark)
    
    run_evaluation_suite_full(input_image=input_image, 
                              embedding_method=embedding_method,
                              tamper_detection_method=tamper_detection_method,
                              recovery_method=recovery_method, n_timing_iterations=100,
                              technique_name=technique_name, image_name=image_name, format=format,
                              do_benchmarking=do_benchmarking, do_attack_suite=do_attack_suite)


def evaluate_svd(input_image, image_name, format=None, do_benchmarking=True, do_attack_suite=True):
    # Housekeeping
    technique_name = "svd"
    key = 15

    def embedding_method(img):
        return embed_svd(input_img=img, key=key)
    
    def tamper_detection_method(img):
        tamper_mask_fs, tampered_pixels, tampered_blocks, extracted_recovery_pixels = tamper_detection_svd(input_img=img, key=key)
        
        kwargs = {"tampered_pixels": tampered_pixels, "tampered_blocks": tampered_blocks,
                  "extracted_recovery_pixels": extracted_recovery_pixels}

        return tamper_mask_fs, kwargs

    def recovery_method(img, tampered_pixels, tampered_blocks, extracted_recovery_pixels):
        return recovery_svd(input_img=img, key=key, tampered_pixels=tampered_pixels,
                            tampered_blocks=tampered_blocks, extracted_recovery_pixels=extracted_recovery_pixels)
    
    run_evaluation_suite_full(input_image=input_image, 
                              embedding_method=embedding_method,
                              tamper_detection_method=tamper_detection_method,
                              recovery_method=recovery_method, n_timing_iterations=100,
                              technique_name=technique_name, image_name=image_name, format=format,
                              do_benchmarking=do_benchmarking, do_attack_suite=do_attack_suite)

def evaluate_molina_garcia(input_image, image_name, format=None, do_benchmarking=True, do_attack_suite=True):
    # Housekeeping
    technique_name = "molina_garcia"
    key_r = secrets.randbits(128)
    key_g = secrets.randbits(128)
    key_b = secrets.randbits(128)

    def embedding_method(img):
        return embed_molina_garcia(img=img, key_r=key_r, key_g=key_g, key_b=key_b)
    
    def tamper_detection_method(img):
        tamper_mask_fs, tamper_mask, tamper_mask_3d, rec_y_bits, rec_cr_bits, rec_cb_bits = tamper_detection_molina_garcia(
            img=img, key_r=key_r, key_g=key_g, key_b=key_b)
    
        
        kwargs = {"tamper_mask": tamper_mask, "tamper_mask_3d": tamper_mask_3d,
                  "rec_y_bits": rec_y_bits, "rec_cr_bits": rec_cr_bits, "rec_cb_bits": rec_cb_bits}

        return tamper_mask_fs, kwargs

    def recovery_method(img, tamper_mask, tamper_mask_3d, rec_y_bits, rec_cr_bits, rec_cb_bits):
        return recovery_molina_garcia(img=img, key_r=key_r, key_g=key_g, key_b=key_b,
                                      tamper_mask=tamper_mask, tamper_mask_3d=tamper_mask_3d,
                                      rec_y_bits=rec_y_bits, rec_cr_bits=rec_cr_bits, rec_cb_bits=rec_cb_bits)
    
    run_evaluation_suite_full(input_image=input_image, 
                              embedding_method=embedding_method,
                              tamper_detection_method=tamper_detection_method,
                              recovery_method=recovery_method, n_timing_iterations=100,
                              technique_name=technique_name, image_name=image_name, format=format,
                              do_benchmarking=do_benchmarking, do_attack_suite=do_attack_suite)


def evaluate_median(input_image, image_name, format=None, do_benchmarking=True, do_attack_suite=True):
    # Housekeeping
    technique_name = "median"
    seed = secrets.randbits(128)

    def embedding_method(img):
        return embed_median(original_image=img, seed=seed, verbose=False)
    
    def tamper_detection_method(img):
        tamper_mask_fs, recovered_image = tamper_detection_median(image=img, seed=seed, do_post_processing=False)
        
        kwargs = {"recovered_image": recovered_image}

        return tamper_mask_fs, kwargs

    def recovery_method(img, recovered_image):
        return recovery_median(image=img, recovered_image=recovered_image)
    
    run_evaluation_suite_full(input_image=input_image, 
                              embedding_method=embedding_method,
                              tamper_detection_method=tamper_detection_method,
                              recovery_method=recovery_method, n_timing_iterations=100,
                              technique_name=technique_name, image_name=image_name, format=format,
                              do_benchmarking=do_benchmarking, do_attack_suite=do_attack_suite)

def evaluate_sisaudia(input_image, image_name, format=None, do_benchmarking=True, do_attack_suite=True):
    # Housekeeping
    technique_name = "sisaudia"

    def embedding_method(img):
        return embed_sisaudia(original_image=img, verbose=False)
    
    def tamper_detection_method(img):
        tampering_mask, recovered_img = tamper_detection_sisaudia(input_img=img)
        
        kwargs = {"recovered_img": recovered_img}

        return tampering_mask, kwargs

    def recovery_method(img, recovered_img):
        return recovery_sisaudia(input_image=img, recovered_img=recovered_img)
    
    run_evaluation_suite_full(input_image=input_image, 
                              embedding_method=embedding_method,
                              tamper_detection_method=tamper_detection_method,
                              recovery_method=recovery_method, n_timing_iterations=100,
                              technique_name=technique_name, image_name=image_name, format=format,
                              do_benchmarking=do_benchmarking, do_attack_suite=do_attack_suite)

class WatermarkingMethod(Enum):
    DoubleDWTFast = "double_dwt_fast"
    LimitedFreqLo = "limited_freq_lo"
    # HiLoFreq = "hi_lo_freq"
    HiLoSpatial = "hi_lo_spatial"
    Median = "median"
    SVD = "svd"
    MolinaGarcia = "molina_garcia"
    Sisaudia = "sisaudia"
    Grayscale = "grayscale"


def get_evaluation_function(method: WatermarkingMethod):
    if method == WatermarkingMethod.DoubleDWTFast:
        return evaluate_double_dwt_fast
    # elif method == WatermarkingMethod.HiLoFreq:
    #     return evaluate_hi_lo_freq
    elif method == WatermarkingMethod.HiLoSpatial:
        return evaluate_hi_lo_spatial
    elif method == WatermarkingMethod.Median:
        return evaluate_median
    elif method == WatermarkingMethod.SVD:
        return evaluate_svd
    elif method == WatermarkingMethod.MolinaGarcia:
        return evaluate_molina_garcia
    elif method == WatermarkingMethod.Sisaudia:
        return evaluate_sisaudia
    elif method == WatermarkingMethod.Grayscale:
        return evaluate_grayscale
    else:
        raise ValueError


def load_image(image_name: str, image_filename: str = None):
    current_directory = os.path.dirname(__file__)
    if image_filename is None:
        img_path = os.path.join(current_directory, 'Images/{}.tiff'.format(image_name))
    else:
        img_path = os.path.join(current_directory, 'Images', image_filename)
    img = imageio.imread(img_path)

    return img, img_path


def do_evaluation(method: WatermarkingMethod, image_name: str, image_filename: str = None, do_benchmarking=True, do_attack_suite=True):
    img, img_path = load_image(image_name=image_name, image_filename=image_filename)
    get_evaluation_function(method=method)(input_image=img, image_name=image_name, format=os.path.splitext(img_path)[-1], do_benchmarking=do_benchmarking, do_attack_suite=do_attack_suite)


def run_all_evaluations():
    for method in WatermarkingMethod:
        if not method == WatermarkingMethod.LimitedFreqLo:
            for image_name in ["Peppers", "Baboon", "Sailboat", "Airplane", "Car"]:
                do_evaluation(method=method, image_name=image_name)

def run_all_evaluations_1024():
    methods = [WatermarkingMethod.HiLoSpatial, WatermarkingMethod.DoubleDWTFast, WatermarkingMethod.Sisaudia, WatermarkingMethod.Median, WatermarkingMethod.MolinaGarcia, WatermarkingMethod.Grayscale, WatermarkingMethod.SVD]
    for method in methods:
        do_evaluation(method=method, image_name="visdrone_1024", image_filename="visdrone_1024.jpg")
        print("Finished", method)

def run_all_evaluations_fullhd():
    methods = [WatermarkingMethod.HiLoSpatial, WatermarkingMethod.DoubleDWTFast, WatermarkingMethod.Sisaudia, WatermarkingMethod.Median, WatermarkingMethod.MolinaGarcia, WatermarkingMethod.Grayscale, WatermarkingMethod.SVD]
    for method in methods:
        do_evaluation(method=method, image_name="visdrone_1920x1080", image_filename="visdrone.jpg", do_benchmarking=True, do_attack_suite=False)
        print("Finished", method)


def run_all_for_images_paper():
    # methods = [WatermarkingMethod.Sisaudia, WatermarkingMethod.MolinaGarcia, WatermarkingMethod.SVD]
    # methods = [WatermarkingMethod.HiLoSpatial]
    methods = [WatermarkingMethod.HiLoSpatial, WatermarkingMethod.DoubleDWTFast, WatermarkingMethod.Sisaudia, WatermarkingMethod.Median, WatermarkingMethod.MolinaGarcia, WatermarkingMethod.Grayscale, WatermarkingMethod.SVD]
    for method in methods:
        do_evaluation(method=method, image_name="Peppers", do_benchmarking=False, do_attack_suite=True)
        print("Finished", method)


def rerun_one_method(method: WatermarkingMethod):
    for image_name in ["Peppers", "Baboon", "Sailboat", "Airplane", "Car"]:
        do_evaluation(method=method, image_name=image_name)


if __name__ == "__main__":
    # print("Grayscale")
    # rerun_one_method(WatermarkingMethod.Grayscale)
    # print("HiLoSpatial")
    # rerun_one_method(WatermarkingMethod.HiLoSpatial)
    # print("DoubleDWTFast")
    # rerun_one_method(WatermarkingMethod.DoubleDWTFast)
    # print("SVD")
    # rerun_one_method(WatermarkingMethod.SVD)
    # print("MolinaGarcia")
    # rerun_one_method(WatermarkingMethod.MolinaGarcia)
    # print("Median")
    # rerun_one_method(WatermarkingMethod.Median)
    # print("Sisaudia")
    # rerun_one_method(WatermarkingMethod.Sisaudia)
    # run_all_evaluations_1024()
    # run_all_for_images_paper()
    run_all_evaluations_1024()
    run_all_evaluations_fullhd()