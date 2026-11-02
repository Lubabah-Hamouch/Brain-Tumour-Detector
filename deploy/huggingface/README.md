---
title: Brain Tumour MRI API
emoji: 🧠
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8010
startup_duration_timeout: 1h
license: other
short_description: Brain tumour MRI segmentation and classification API
pinned: false
---

# Brain Tumour MRI API

The API behind the brain tumour MRI web app: 3D tumour segmentation of BraTS-style multimodal MRI
(a 5-fold 3D U-Net ensemble) and tumour-type classification of 2D MRI slices (EfficientNetV2-S
with Grad-CAM). It runs on this Space's CPU.

- **Interactive docs:** `/docs` on this Space's URL.
- **Health:** `/api/health` reports the loaded models and whether the sample results are ready.
- **Code, results and documentation:** https://github.com/Lubabah-Hamouch/Brain-Tumour-Detector

**Start-up:** the container downloads the trained weights (about 410 MB) from the repository's
`weights-v1` release, then loads the six models. After a period without visitors the Space
sleeps, and the next request wakes it, which takes a few minutes.

**Speed:** on this CPU, classifying an image takes a few seconds and segmenting a study takes
several minutes. The sample study and the five sample images are answered at once from results
computed in advance.

**Research use only. Not a medical device.**

## Data and terms of use

This Space offers samples from two datasets and serves weights trained on them:

- five held-out test images from the Brain Tumor MRI Dataset;
- for one held-out BraTS 2021 test patient (`BraTS2021_00003`), the model's results only: the
  predicted label map, viewer slice images, 3D surfaces and volumes.

The patient's raw MRI volumes are not part of this Space and are not served. All of this remains
under each dataset's terms:

- **BraTS 2021 Task 1** (RSNA-ASNR-MICCAI Brain Tumor Segmentation Challenge), Synapse ID
  syn25829067. Cite U. Baid et al., arXiv:2107.02314 (2021); B. H. Menze et al., IEEE TMI
  34(10), 1993–2024 (2015); S. Bakas et al., Scientific Data 4, 170117 (2017).
- **Brain Tumor MRI Dataset** (Kaggle `masoudnickparvar/brain-tumor-mri-dataset`, CC BY 4.0).

The code is MIT-licensed; see the repository.
