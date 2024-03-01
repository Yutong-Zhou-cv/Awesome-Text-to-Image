# <p align=center>🧸Datasets</p>

  * - [x] [Oxford-102 Flower](#head-flower)
  * - [x] [Caltech-UCSD Bird (CUB)](#head-cub)
  * - [x] [MS-COCO](#head-coco)
  * - [x] [Multi-Modal-CelebA-HQ](#head-mmdata)
  * - [x] [CelebA-Dialog](#head-celebad)
  * - [x] [FFHQ-Text](#head-ffhqtext)
  * - [x] [CelebAText-HQ](#head-celebatext)
  * - [x] [DeepFashion-MultiModal](#head-deepfashion)
  * - [x] [ANNA](#head-anna)
  * - [x] [Bento800-Text](#head-bento)
  * - [ ] [Others](#head-others)


## Details
* <span id="head-flower"> **Oxford-102 Flower** </span>

  Oxford-102 Flower is a 102-category dataset, consisting of 102 flower categories. The flowers are chosen to be flowers commonly occurring in the United Kingdom. The images have large scale, pose and light variations. 
  * **Detailed information (Images):**  ⇒ [[Paper](http://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/nilsback08.pdf)] [[Website](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)]
    * Number of different categories: 102 (**Training**: 82 categories. **Testing**: 20 categories.)
    * Number of flower images: 8,189
  * **Detailed information (Text Descriptions):**  ⇒ [[Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Reed_Learning_Deep_Representations_CVPR_2016_paper.pdf)] [[Download](https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view?usp=sharing&resourcekey=0-Av8zFbeDDvNcF1sSjDR32w)]
    * Descriptions per image: 10 Captions

* <span id="head-cub"> **Caltech-UCSD Bird(CUB)** </span>

  Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations.
  * **Detailed information (Images):**  ⇒ [[Paper](http://www.vision.caltech.edu/visipedia/papers/CUB_200_2011.pdf)] [[Website](http://www.vision.caltech.edu/datasets/cub_200_2011/)]
    * Number of different categories: 200 (**Training**: 150 categories. **Testing**: 50 categories.)
    * Number of bird images: 11,788
    * Annotations per image: 15 Part Locations, 312 Binary Attributes, 1 Bounding Box, Ground-truth Segmentation
  * **Detailed information (Text Descriptions):**  ⇒ [[Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Reed_Learning_Deep_Representations_CVPR_2016_paper.pdf)] [[Website](https://drive.google.com/file/d/0B0ywwgffWnLLZW9uVHNjb2JmNlE/view)]
    * Descriptions per image: 10 Captions

* <span id="head-coco"> **MS-COCO** </span>

  COCO is a large-scale object detection, segmentation, and captioning dataset.
  * **Detailed information (Images):**  ⇒ [[Paper](https://arxiv.org/pdf/1405.0312.pdf)] [[Website](https://cocodataset.org/#overview)]
    * Number of different categories: 91
    * Number of images: 120k (**Training**: 80k. **Testing**: 40k.)
  * **Detailed information (Text Descriptions):** ⇒ [[Paper](https://arxiv.org/pdf/1405.0312.pdf)] [[Download](https://drive.google.com/file/d/1GOEl9lxgSsWUWOXkZZrch08GgPADze7U/view?usp=sharing)]
    * Descriptions per image: 5 Captions
  
* <span id="head-mmdata"> **Multi-Modal-CelebA-HQ** </span>

  Multi-Modal-CelebA-HQ is a large-scale face image dataset for text-to-image-generation, text-guided image manipulation, sketch-to-image generation, GANs for face generation and editing, image caption, and VQA.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://arxiv.org/pdf/2012.03308.pdf)] [[Website](https://github.com/weihaox/Multi-Modal-CelebA-HQ-Dataset)] [[Download](https://drive.google.com/drive/folders/1eVrGKfkbw7bh9xPcX8HJa-qWQTD9aWvf)]
    * Number of images (from Celeba-HQ): 30,000 (**Training**: 24,000. **Testing**: 6,000.)
    * Descriptions per image: 10 Captions
  * **Detailed information (Masks):** 
    * Number of masks (from Celeba-Mask-HQ): 30,000 (512 x 512)
  * **Detailed information (Sketches):** 
    * Number of Sketches: 30,000 (512 x 512)
  * **Detailed information (Image with transparent background):** 
    * Not fully uploaded

* <span id="head-celebad"> **CelebA-Dialog** </span>

  CelebA-Dialog is a large-scale visual-language face dataset. It has two properties:
  (1) Facial images are annotated with **rich fine-grained labels**, which classify one attribute into multiple degrees according to its semantic meaning.
  (2) Accompanied with each image, there are **captions describing** the attributes and a **user request sample**.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Talk-To-Edit_Fine-Grained_Facial_Editing_via_Dialog_ICCV_2021_paper.pdf)] [[Website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Dialog.html)] [[Download](https://github.com/yumingj/Talk-to-Edit)]
    * Number of identities: 10,177
    * Number of images: 202,599 
    * 5 fine-grained attributes annotations per image: Bangs, Eyeglasses, Beard, Smiling, and Age


* <span id="head-ffhqtext"> **FFHQ-Text** </span>

  FFHQ-Text is a small-scale face image dataset with large-scale facial attributes, designed for text-to-face generation & manipulation, text-guided facial image manipulation, and other vision-related tasks.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3481026)] [[Website](https://github.com/Yutong-Zhou-cv/FFHQ-Text_Dataset)] [[Download](https://forms.gle/f7oMXD3g9BgdgEUd7)]
    * Number of images (from FFHQ): 760 (**Training**: 500. **Testing**: 260.)
    * Descriptions per image: 9 Captions
    * 13 multi-valued facial element groups from coarse to fine.
  * **Detailed information (BBox):** ⇒ [[Website](https://www.robots.ox.ac.uk/~vgg/software/via/)]

* <span id="head-celebAtext"> **CelebAText-HQ** </span>

  CelebAText-HQ is a large-scale face image dataset with large-scale facial attributes, designed for text-to-face generation.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475391)] [[Website](https://github.com/cripac-sjx/SEA-T2F)] [[Download](https://drive.google.com/drive/folders/1IAb_iy6-soEGQWhbgu6cQODsIUJZpahC)]
    * Number of images (from Celeba-HQ): 15010 (**Training**: 13,710. **Testing**: 1300.)
    * Descriptions per image: 10 Captions

* <span id="head-deepfashion"> **DeepFashion-MultiModal** </span>
  
  DeepFashion-MultiModal is a large-scale high-quality human dataset. Human images are annotated with **rich multi-modal labels**, including human parsing labels, keypoints, densepose, fine-grained attributes and textual descriptions.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://arxiv.org/pdf/2205.15996.pdf)] [[Website](https://github.com/yumingj/DeepFashion-MultiModal)] [[Download](https://drive.google.com/drive/folders/1An2c_ZCkeGmhJg0zUjtZF46vyJgQwIr2?usp=sharing)]
    * Number of images: 44,096, including 12,701 full-body images
    * Descriptions per image: 1 Caption

* <span id="head-anna"> **ANNA** </span>

  ANNA is an Abstractive News captioNs dAtaset extracted from online news articles in a variety of different contexts. The generated images are judged on the basis of contextual relevance, visual quality, and perceptual similarity to ground-truth image-caption pairs.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://arxiv.org/abs/2301.02160)] [[Download](https://github.com/aashish2000/ANNA)]
    * Number of image-text pairs (from The New York Times): 29625 (**Training**: 17897. **Validation**: 1622. **Testing**: 1649.)

* <span id="head-bento"> **Bento800-Text** </span>

  Bento800 is the first manually annotated synthetic box lunch presentation dataset with diverse annotations(BBox, Segmentation, Labels... ) for novel aesthetic box lunch presentation design.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://dl.acm.org/doi/10.1145/3552485.3554935)] [[Website](https://github.com/Yutong-Zhou-cv/Bento800_Dataset)] [[Download](https://drive.google.com/drive/folders/1_VvAbIzeuVew4fa98CcE11mB9SoAI3q-?usp=sharing)]
    * Number of images: 800 (**Training**: 766. **Testing**: 34.)
    * Descriptions per image: 9 Captions

* <span id="head-others"> **Others** </span>
    * (arXiv preprint 2023) **AGIQA-3K** [[Paper](https://arxiv.org/abs/2306.04717)] [[Dataset](https://github.com/lcysyzxdxc/AGIQA-3k-Database/tree/main)]
    * (CVPR 2023) [💬 Video] **CelebV-Text** [[Paper](https://arxiv.org/abs/2303.14717)] [[Github](https://github.com/celebv-text/CelebV-Text)] [[Project](https://celebv-text.github.io/)]
    * (ECCV 2022) [💬 Video] **CelebV-HQ** [[Paper](https://arxiv.org/abs/2207.12393)] [[Github](https://github.com/celebv-hq/CelebV-HQ)] [[Project](https://celebv-hq.github.io/)]

**[       «🎯Back To Top»       ](#)**

