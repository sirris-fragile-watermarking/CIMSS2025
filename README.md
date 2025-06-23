# CIMSS 2025 code

This repository contains the code that goes alongside the below publication in the `5th International Workshop on Critical Infrastructure and Manufacturing System Security (CIMSS)` at the Applied Cryptography and Network Security (ACNS) conference in Munich in June 2025.

Please use the below reference when using this code:
```
Fast and robust fragile watermarking enabling real-time self-recovery for UAS,
Applied Cryptography and Network Security Workshops,
Laurens Le Jeune, Anna Hristoskova, Farhad Aghili
```

Or use the below bibtex:
```
@InProceedings{ToBeCompleted,
    author={Le Jeune, Laurens and Hristoskova, Anna and Aghili, Farhad},
    editor={},
    title={Fast and robust fragile watermarking enabling real-time self-recovery for {UAS}},
    booktitle={Applied Cryptography and Network Security Workshops},
    year={2025},
    publisher={Springer International Publishing},
    address={Cham},
    pages={},
    isbn={}
}
```

## Overview
The enclosed code can be used to reproduce our experiments. We implement seven methods:

* `custom_spatial_based.py` is the HiLoSpatial approach;
* `grayscale_recovery.py` is the HiResSpatial approach;
* `lbp_based.py` is code based on the work of Sisaudia et al.[1];
* `molina_garcia.py` is based on the work of Molina-Garcia et al. [2];
* `median_based.py` is based on the work of Rajput et al. [3];
* `svd_based.py` is based on the work of Shebab et al. [4];
* `functions_without_config.py`, `own_functions_without_config.py` and `original_code_interface.py` are based on the work of Bouarroudj et al. [5], as taken from their [Github](https://github.com/Riadh-Bouarroudj/Fragile-image-watermarking-with-recovery).

Note that we include a copy of Bouarroudje et al.'s code in the folder `bouarroudj_original_code`, including a copy of the Apache 2.0 license. We also provide a copy of the license in our modified files that are based on their original code.

## Reproducing experiments
The following modules can be used to reproduce our experiments:

* `detection_coverage.py` reproduces the coverage results of Tab. 2;
* `benchmarking.py` reproduces the actual watermark embedding, tamper detection and recovery experiments (Tab. 3, 4 ,5);
* `produce_license_plate_example.py` generates the images used in Fig. 6;

The `export.py` and `export_paper.py` modules contain tools to parse the generated experimental data, as stored in the `benchmarking` folder.
The `demo` folder is used as the target to store the license plate example images.

## References
* [1] Sisaudia, V., Vishwakarma, V.P.: Approximate regeneration of image using fragile watermarking for tamper detection and recovery in real time. Multimedia Tools and Applications 83(25), 66299–66318 (Jul 2024).
* [2] Molina-Garcia, J., Garcia-Salgado, B.P., Ponomaryov, V., Reyes-Reyes, R., Sadovnychiy, S., Cruz-Ramos, C.: An effective fragile watermarking scheme for color image tampering detection and self-recovery. Signal Processing: Image Communication 81, 115725 (2020).
* [3] Rajput, V., Ansari, I.A.: Image tamper detection and self-recovery using multiple median watermarking. Multimedia Tools and Applications 79(47), 35519–3553 (Dec 2020).
* [4] Shehab, A., Elhoseny, M., Muhammad, K., Sangaiah, A.K., Yang, P., Huang, H., Hou, G.: Secure and robust fragile watermarking scheme for medical images. IEEE Access 6, 10269–10278 (2018).
* [5] Bouarroudj, R., Souami, F., Belalla, F.Z.: Reversible fragile watermarking for medical image authentication in the frequency domain. In: 2023 2nd IC2EM. vol. 1, pp. 1–6 (2023).