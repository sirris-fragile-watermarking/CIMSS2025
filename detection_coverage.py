import numpy as np
import secrets

from help_functions import BinaryClassification
from benchmarking import WatermarkingMethod
from custom_spatial_based import embed_watermark as embed_hilo
from custom_spatial_based import __tamper_detection_for_test as detect_hilo

from grayscale_recovery import watermarking as embed_gray
from grayscale_recovery import __tamper_detection_for_test as detect_gray

import multiprocessing as mp
from time import time


def compute_for_edge(edge_len, n_its, size_x, size_y, watermarked, n_channels, rng, detect, seed):
        base_results = []
        grow_results = []

        for i in range(n_its):
            
            attacked = watermarked.copy()
            tamper_mask = np.zeros((size_x, size_y), dtype=np.uint8)
            x_start = rng.integers(low=0, high=size_x - edge_len, endpoint=False)
            y_start = rng.integers(low=0, high=size_y - edge_len, endpoint=False)
            tampering = rng.integers(low=0, high=256, endpoint=False, size=(edge_len, edge_len, n_channels)).astype(np.uint8)
            attacked[x_start:x_start+edge_len, y_start:y_start + edge_len] = tampering
            tamper_mask[x_start:x_start+edge_len, y_start:y_start + edge_len] = 1
            grow, base = detect(attacked, seed=seed)

            grow_1d = grow.reshape(-1)
            base_1d = base.reshape(-1)
            true_1d = tamper_mask.reshape(-1)
            
            base_coverage = BinaryClassification(y_true=true_1d, y_pred=base_1d).recall()
            grow_coverage = BinaryClassification(y_true=true_1d, y_pred=grow_1d).recall()

            base_results.append(base_coverage)
            grow_results.append(grow_coverage)
        
        base_coverage = float(np.round(np.average(base_results), decimals=4))
        grow_coverage = float(np.round(np.average(grow_results), decimals=4))
        print("{}: Base coverage: {}".format(edge_len, base_coverage))
        print("{}: Grow coverage: {}".format(edge_len, grow_coverage))

        return (edge_len, base_coverage, grow_coverage)


def test_tamper_detection_coverage_mp(method: WatermarkingMethod, n_its=1000, n_channels=3):
    if method == WatermarkingMethod.HiLoSpatial:
        embed = embed_hilo
        detect = detect_hilo

    elif method == WatermarkingMethod.Grayscale:
        embed = embed_gray
        detect = detect_gray
    
    seed = secrets.randbits(128)

    square_edges = [1, 2, 4, 16, 32, 128, 256]

    # Watermark a zero-image
    size_x = 512
    size_y = 512
    
    image = np.zeros((size_x, size_y, n_channels), dtype=np.uint8)

    watermarked = embed(image, seed=seed)
    rng = np.random.default_rng(seed=seed + 1)


    with mp.Pool(processes=len(square_edges)) as pool:
        arguments = [[edge_len, n_its, size_x, size_y, watermarked, n_channels, rng, detect, seed] for edge_len in square_edges]
        
        t1 = time()
        async_results = pool.starmap_async(compute_for_edge, arguments)
        results = async_results.get()
        t2 = time()
        
        print("Computation took {t:.4} seconds".format(t=t2 - t1))
        print(results)
        


def test_tamper_detection_coverage(method: WatermarkingMethod, n_its=1000, n_channels=3):
    
    if method == WatermarkingMethod.HiLoSpatial:
        embed = embed_hilo
        detect = detect_hilo

    elif method == WatermarkingMethod.Grayscale:
        embed = embed_gray
        detect = detect_gray
    
    seed = secrets.randbits(128)

    square_edges = [1, 2, 4, 16, 32, 128, 256]

    # Watermark a zero-image
    size_x = 512
    size_y = 512
    
    image = np.zeros((size_x, size_y, n_channels), dtype=np.uint8)

    watermarked = embed(image, seed=seed)
    rng = np.random.default_rng(seed=seed + 1)

    max_edge_len = len(str(max(square_edges)))
    max_i_len = len(str(n_its))

    for edge_len in square_edges:
        base_results = []
        grow_results = []

        for i in range(n_its):
            
            attacked = watermarked.copy()
            tamper_mask = np.zeros((size_x, size_y), dtype=np.uint8)
            x_start = rng.integers(low=0, high=size_x - edge_len, endpoint=False)
            y_start = rng.integers(low=0, high=size_y - edge_len, endpoint=False)
            tampering = rng.integers(low=0, high=256, endpoint=False, size=(edge_len, edge_len, n_channels)).astype(np.uint8)
            attacked[x_start:x_start+edge_len, y_start:y_start + edge_len] = tampering
            tamper_mask[x_start:x_start+edge_len, y_start:y_start + edge_len] = 1
            grow, base = detect(attacked, seed=seed)

            grow_1d = grow.reshape(-1)
            base_1d = base.reshape(-1)
            true_1d = tamper_mask.reshape(-1)
            
            base_coverage = BinaryClassification(y_true=true_1d, y_pred=base_1d).recall()
            grow_coverage = BinaryClassification(y_true=true_1d, y_pred=grow_1d).recall()

            base_results.append(base_coverage)
            grow_results.append(grow_coverage)

            if i % 10 == 0:
                print("Edge: {edge: <{l1}}: {i: >{l2}} its".format(edge=edge_len, l1=max_edge_len, i=i, l2=max_i_len), end="\r")

        print("For edge_len {} ({} pixels):".format(edge_len, edge_len**2))
        print("Base coverage: {}".format(np.round(np.average(base_results), decimals=4)))
        print("Grow coverage: {}".format(np.round(np.average(grow_results), decimals=4)))



if __name__ == "__main__":
    # test_tamper_detection_coverage_mp(method=WatermarkingMethod.HiLoSpatial, n_its=100000, n_channels=3)
    # test_tamper_detection_coverage_mp(method=WatermarkingMethod.HiLoSpatial, n_its=100000, n_channels=1)
    test_tamper_detection_coverage_mp(method=WatermarkingMethod.Grayscale, n_its=100000, n_channels=3)