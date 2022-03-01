# <p align=center>Awesome Text📝-to-Image🌇</p>
<!--# <p align=center>`Awesome Text📝-to-Image🌇`</p>-->

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) ![Visitors](https://visitor-badge.glitch.me/badge?page_id=Yutong-Zhou-cv/Awesome-Text-to-Image) ![GitHub stars](https://img.shields.io/github/stars/Yutong-Zhou-cv/Awesome-Text-to-Image.svg?color=red) 

A collection of resources on text-to-image synthesis task.

## <span id="head-content"> *Content* </span>
* - [x] [1. Description](#head1)

* - [x] [2. Quantitative Evaluation Metrics](#head2)
  * [Inception Score (IS)](#head-IS)
  * [Fréchet Inception Distance (FID)](#head-FID)  
  * [R-precision](#head-R)
  * [L<sub>2</sub> error](#head-L2)
  
* - [x] [3. Datasets](#head3)  
  * [Caltech-UCSD Bird (CUB)](#head-CUB)
  * [Oxford-102 Flower](#head-Flower)
  * [MS-COCO](#head-COCO)
  * [Multi-Modal-CelebA-HQ](#head-Multi-Modal-CelebA-HQ)
  * [FFHQ-Text](#head-FFHQ-Text)

* - [ ] [4. Project](#head4)

* - [ ] [5. Paper With Code](#head5)
  * - [ ] [Survey](#head-Survey)
  * - [ ] [2022](#head-2022)
  * - [ ] [2021](#head-2021)
  * - [x] [2020](#head-2020)
  * - [x] [2019](#head-2019)
  * - [x] [2018](#head-2018)
  * - [x] [2017](#head-2017)
  * - [x] [2016](#head-2016)
  
* - [ ] [5. Other Related Works](#head5)
  * - [ ] [⭐Multimodality⭐](#head-MM)
  * - [ ] [Label-set → Semantic maps](#head-L2S)
  * - [ ] [Text+Image → Image](#head-TI2I)
  * - [ ] [Layout → Image](#head-L2I)
  * - [ ] [Speech → Image](#head-S2I)
  * - [ ] [Text → Visual Retrieval](#head-T2VR)
  * - [ ] [Text → Video](#head-T2V)

* [*Contact Me*](#head6)

 ## <span id="head1"> *1. Description* </span>

* In the last few decades, the fields of Computer Vision (CV) and Natural Language Processing (NLP) have been made several major technological breakthroughs in deep learning research. Recently, researchers appear interested in combining semantic information and visual information in these traditionally independent fields. 
A number of studies have been conducted on the text-to-image synthesis techniques that transfer input textual description (keywords or sentences) into realistic images.

* Papers, codes and datasets for the text-to-image task are available here.

>🐌 Markdown Format:
> * (Conference/Journal Year) **Title**, First Author et al. [[Paper](URL)] [[Code](URL)] [[Project](URL)]

 ## <span id="head2"> *2. Quantitative Evaluation Metrics* </span> [       «🎯Back To Top»       ](#)

* <span id="head-IS"> Inception Score (IS) </span> [[Paper](https://arxiv.org/pdf/1606.03498.pdf)] [[Python Code (Pytorch)](https://github.com/sbarratt/inception-score-pytorch)] [[Python Code (Tensorflow)](https://github.com/taki0112/GAN_Metrics-Tensorflow)] [[Ref.Code(AttnGAN)](https://github.com/taoxugit/AttnGAN)]

* <span id="head-FID"> Fréchet Inception Distance (FID) </span> [[Paper](https://papers.nips.cc/paper/7240-gans-trained-by-a-two-time-scale-update-rule-converge-to-a-local-nash-equilibrium.pdf)] [[Python Code (Pytorch)](https://github.com/mseitzer/pytorch-fid)] [[Python Code (Tensorflow)](https://github.com/taki0112/GAN_Metrics-Tensorflow)] [[Ref.Code(DM-GAN)](https://github.com/MinfengZhu/DM-GAN)]

* <span id="head-R"> R-precision </span> [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf)] [[Ref.Code(CPGAN)](https://github.com/dongdongdong666/CPGAN)]

* <span id="head-L2"> L<sub>2</sub> error </span> [[Paper](https://papers.nips.cc/paper/7290-text-adaptive-generative-adversarial-networks-manipulating-images-with-natural-language.pdf)]

## <span id="head3"> *3. Datasets* </span> [       «🎯Back To Top»       ](#)

* <span id="head-CUB"> **Caltech-UCSD Bird(CUB)** </span>

  Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations.
  * **Detailed information (Images):**  ⇒ [[Paper](http://www.vision.caltech.edu/visipedia/papers/CUB_200_2011.pdf)] [[Website](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)]
    * Number of different categories: 200 (**Training**: 150 categories. **Testing**: 50 categories.)
    * Number of bird images: 11,788
    * Annotations per image: 15 Part Locations, 312 Binary Attributes, 1 Bounding Box, Ground-truth Segmentation
  * **Detailed information (Text Descriptions):**  ⇒ [[Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Reed_Learning_Deep_Representations_CVPR_2016_paper.pdf)] [[Website](https://drive.google.com/file/d/0B0ywwgffWnLLZW9uVHNjb2JmNlE/view)]
    * Descriptions per image: 10 Captions
    
* <span id="head-Flower"> **Oxford-102 Flower** </span>

  Oxford-102 Flower is a 102 category dataset, consisting of 102 flower categories. The flowers are chosen to be flower commonly occurring in the United Kingdom. The images have large scale, pose and light variations. 
  * **Detailed information (Images):**  ⇒ [[Paper](http://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/nilsback08.pdf)] [[Website](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)]
    * Number of different categories: 102 (**Training**: 82 categories. **Testing**: 20 categories.)
    * Number of flower images: 8,189
  * **Detailed information (Text Descriptions):**  ⇒ [[Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Reed_Learning_Deep_Representations_CVPR_2016_paper.pdf)] [[Download](https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view?usp=sharing&resourcekey=0-Av8zFbeDDvNcF1sSjDR32w)]
    * Descriptions per image: 10 Captions
    
* <span id="head-COCO"> **MS-COCO** </span>

  COCO is a large-scale object detection, segmentation, and captioning dataset.
  * **Detailed information (Images):**  ⇒ [[Paper](https://arxiv.org/pdf/1405.0312.pdf)] [[Website](https://cocodataset.org/#overview)]
    * Number of different categories: 91
    * Number of images: 120k (**Training**: 80k. **Testing**: 40k.)
  * **Detailed information (Text Descriptions):** ⇒ [[Paper](https://arxiv.org/pdf/1405.0312.pdf)] [[Download](https://drive.google.com/file/d/1GOEl9lxgSsWUWOXkZZrch08GgPADze7U/view?usp=sharing)]
    * Descriptions per image: 5 Captions
    
* <span id="head-Multi-Modal-CelebA-HQ"> **Multi-Modal-CelebA-HQ** </span>

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

* <span id="head-FFHQ-Text"> **FFHQ-Text** </span>

  FFHQ-Text is a small-scale face image dataset with large-scale facial attributes, designed for text-to-face generation & manipulation, text-guided facial image manipulation, and other vision-related tasks.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3481026)] [[Website](https://github.com/Yutong-Zhou-cv/FFHQ-Text_Dataset)] [[Download](https://forms.gle/f7oMXD3g9BgdgEUd7)]
    * Number of images (from FFHQ): 760 (**Training**: 500. **Testing**: 260.)
    * Descriptions per image: 9 Captions
    * 13 multi-valued facial element groups from coarse to fine.
  * **Detailed information (BBox):** ⇒ [[Website](https://www.robots.ox.ac.uk/~vgg/software/via/)]

## <span id="head4"> *4. Project* </span>
* **Aphantasia**. [[Github](https://github.com/eps696/aphantasia)] 
    * >This is a text-to-image tool, part of the [artwork](https://computervisionart.com/pieces2021/aphantasia/) of the same name. ([Aphantasia](https://en.wikipedia.org/wiki/Aphantasia) is the inability to visualize mental images, the deprivation of visual dreams.)
* **Text2Art**. [[Try it now!](https://text2art.com/)] [[Github](https://github.com/mfrashad/text2art)] [[Blog](https://towardsdatascience.com/how-i-built-an-ai-text-to-art-generator-a0c0f6d6f59f)] 
    * >Text2Art is an AI-powered art generator based on VQGAN+CLIP that can generate all kinds of art such as pixel art, drawing, and painting from just text input. 
* **Survey Text Based Image Synthesis** [[Blog](https://hackmd.io/@prajwalsingh/imagesynthesis#) (2021)]

## <span id="head5"> *5. Paper With Code* </span>

* <span id="head-Survey"> **Survey**  </span> **[       «🎯Back To Top»       ](#)**
    * **Text-to-Image Synthesis: A Comparative Study** [[v1](https://link.springer.com/chapter/10.1007/978-981-16-2275-5_14)(Digital Transformation Technology)] (2021.08) 
    * **A survey on generative adversarial network-based text-to-image synthesis** [[v1](https://www.sciencedirect.com/science/article/pii/S0925231221006111)(Neurocomputing)] (2021.04) 
    * **Adversarial Text-to-Image Synthesis: A Review** [[v1](https://arxiv.org/abs/2101.09983v1)(arXiv)] (2021.01) [[v2](https://www.sciencedirect.com/science/article/pii/S0893608021002823)(Neural Networks)] (2021.08)
    * **A Survey and Taxonomy of Adversarial Neural Networks for Text-to-Image Synthesis** [[v1](https://arxiv.org/pdf/1910.09399.pdf)(arXiv)] (2019.10) 

* <span id="head-2022"> **2022**  </span> **[       «🎯Back To Top»       ](#)**
    * (arXiv preprint 2022) **OptGAN: Optimizing and Interpreting the Latent Space of the Conditional Text-to-Image GANs**, Zhenxing Zhang et al. [[Paper](https://arxiv.org/abs/2202.12929)]
    * ⭐⭐(arXiv preprint 2022) **DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers**, Jaemin Cho et al. [[Paper](https://arxiv.org/abs/2202.04053)] [[Code](https://github.com/j-min/DallEval)] 
    * (IEEE Transactions on Network Science and Engineering) **Neural Architecture Search with a Lightweight Transformer for Text-to-Image Synthesis**, Wei Li et al. [[Paper](https://ieeexplore.ieee.org/abstract/document/9699403)] 
    * (Neurocomputing 2022) **DiverGAN: An Efficient and Effective Single-Stage Framework for Diverse Text-to-Image Generation**, Zhenxing Zhang et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231221018397)]
    * (Knowledge-Based Systems) **CJE-TIG: Zero-shot cross-lingual text-to-image generation by Corpora-based Joint Encoding**, Han Zhang et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S0950705121011138)] 
    * (WACV 2022) **StyleMC: Multi-Channel Based Fast Text-Guided Image Generationand Manipulation**, Umut Kocasarı et al. [[Paper](https://arxiv.org/abs/2112.08493)] [[Project](https://catlab-team.github.io/stylemc/)]

* <span id="head-2021"> **2021**  </span> **[       «🎯Back To Top»       ](#)**
    * (FG 2021) **Generative Adversarial Network for Text-to-Face Synthesis and Manipulation with Pretrained BERT Model**, Yutong Zhou et al. [[Paper](https://ieeexplore.ieee.org/document/9666791)] 
    * (arXiv preprint 2021) **Multimodal Conditional Image Synthesis with Product-of-Experts GANs**, Xun Huang et al. [[Paper](https://arxiv.org/abs/2112.05130)]  [[Project](https://deepimagination.cc/PoE-GAN/)]
         * Text-to-Image, Segmentation-to-Image, *Text+Segmentation/Sketch/Image→Image*, *Sketch+Segmentation/Image→Image*, *Segmentation+Image→Image*
    * (arXiv preprint 2021) **More Control for Free! Image Synthesis with Semantic Diffusion Guidance**, Xihui Liu et al. [[Paper](https://arxiv.org/abs/2112.05744)] [[Project](https://xh-liu.github.io/sdg/)] 
    * (IEEE TCSVT) **RiFeGAN2: Rich Feature Generation for Text-to-Image Synthesis from Constrained Prior Knowledge**, Jun Cheng et al. [[Paper](https://ieeexplore.ieee.org/abstract/document/9656731)] 
    * (ICONIP 2021) **TRGAN: Text to Image Generation Through Optimizing Initial Image**, Liang Zhao et al. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-92307-5_76)] 
    * ⭐(arXiv preprint 2021) **GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models**, Alex Nichol et al. [[Paper](https://arxiv.org/abs/2112.10741)] [[Code](https://github.com/openai/glide-text2im)]
    * (NeurIPS 2021) **Benchmark for Compositional Text-to-Image Synthesis**, Dong Huk Park et al. [[Paper](https://openreview.net/forum?id=bKBhQhPeKaF)]
    * ⭐(arXiv preprint 2021) **FuseDream: Training-Free Text-to-Image Generation with Improved CLIP+GAN Space Optimization**, Xingchao Liu et al. [[Paper](https://arxiv.org/abs/2112.01573)] [[Code](https://github.com/gnobitab/FuseDream)]
    * (arXiv preprint 2021) [💬Evaluation] **TISE: A Toolbox for Text-to-Image Synthesis Evaluation**, Tan M. Dinh et al. [[Paper](https://arxiv.org/abs/2112.01398)] [[Project](https://di-mi-ta.github.io/tise/)]
    * (ICONIP 2021) **Self-Supervised Image-to-Text and Text-to-Image Synthesis**, Anindya Sundar Das et al. [[Paper](https://arxiv.org/abs/2112.04928)]
    * (arXiv preprint 2021) **LAFITE: Towards Language-Free Training for Text-to-Image Generation**, Yufan Zhou et al. [[Paper](https://arxiv.org/abs/2111.13792)] 
    * (arXiv preprint 2021) **Vector Quantized Diffusion Model for Text-to-Image Synthesis**, Shuyang Gu et al. [[Paper](https://arxiv.org/abs/2111.14822)] [[Code](https://github.com/microsoft/vq-diffusion)]
    * ⭐⭐(arXiv preprint 2021) **NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion**, Chenfei Wu et al. [[Paper](https://arxiv.org/pdf/2111.12417.pdf)] [[Code](https://github.com/microsoft/NUWA)]
        * **Multimodal Pretrained Model for Multi-tasks🎄**: Text-To-Image (T2I), Sketch-to-Image (S2I), Image Completion (I2I), Text-Guided Image Manipulation (TI2I), Text-to-Video (T2V), Video Prediction (V2V), Sketch-to-Video (S2V), Text-Guided Video Manipulation (TV2V)
          ![Figure from paper](pic/NUWA.gif)
          > *(From: https://github.com/microsoft/NUWA [2021/11/30])*
    * (arXiv preprint 2021) **DiverGAN: An Efficient and Effective Single-Stage Framework for Diverse Text-to-Image Generation**, Zhenxing Zhang et al. [[Paper](https://arxiv.org/pdf/2111.09267.pdf)] 
    * (Image and Vision Computing) **Transformer models for enhancing AttnGAN based text to image generation**, S. Naveen et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S026288562100189X)]
    * (ACMMM 2021) **R-GAN: Exploring Human-like Way for Reasonable Text-to-Image Synthesis via Generative Adversarial Networks**, Yanyuan Qiao et al. [[Paper](https://dl.acm.org/doi/10.1145/3474085.3475363)]
    * (ACMMM 2021) **Multi-caption Text-to-Face Synthesis: Dataset and Algorithm**, Jianxin Sun et al. [[Paper](https://dl.acm.org/doi/10.1145/3474085.3475391)] [[Code](https://github.com/cripac-sjx/SEA-T2F)]
    * (ACMMM 2021) **Cycle-Consistent Inverse GAN for Text-to-Image Synthesis**, Hao Wang et al. [[Paper](https://dl.acm.org/doi/10.1145/3474085.3475226)]
    * (ACMMM 2021) **Unifying Multimodal Transformer for Bi-directional Image and Text Generation**, Yupan Huang et al. [[Paper](https://dl.acm.org/doi/10.1145/3474085.3481540)] [[Code](https://github.com/researchmm/generate-it)]
    * (ACMMM 2021) **A Picture is Worth a Thousand Words: A Unified System for Diverse Captions and Rich Images Generation**, Yupan Huang et al. [[Paper](https://dl.acm.org/doi/10.1145/3474085.3478561)] [[Code](https://github.com/researchmm/generate-it)]
    * (ACMMM 2021) **Generative Adversarial Network for Text-to-Face Synthesis and Manipulation**, Yutong Zhou. [[Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3481026)]
    * (ICCV 2021) **DAE-GAN: Dynamic Aspect-Aware GAN for Text-to-Image Synthesis**, Shulan Ruan et al. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ruan_DAE-GAN_Dynamic_Aspect-Aware_GAN_for_Text-to-Image_Synthesis_ICCV_2021_paper.pdf)] [[Supp](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Ruan_DAE-GAN_Dynamic_Aspect-Aware_ICCV_2021_supplemental.pdf)] [[Code](https://github.com/hiarsal/DAE-GAN)]
    * (ICIP 2021) **Text To Image Synthesis With Erudite Generative Adversarial Networks**, Zhiqiang Zhang et al. [[Paper](https://ieeexplore.ieee.org/document/9506487)] 
    * (PRCV 2021) **MAGAN: Multi-attention Generative Adversarial Networks for Text-to-Image Generation**, Xibin Jia et al. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-88013-2_26)]
    * (AAAI 2021) **TIME: Text and Image Mutual-Translation Adversarial Networks**, Bingchen Liu et al. [[Paper](https://www.aaai.org/AAAI21Papers/AAAI-1426.LiuB.pdf)] [[arXiv Paper](https://arxiv.org/pdf/2005.13192.pdf)] 
    * (IJCNN 2021) **Text to Image Synthesis based on Multi-Perspective Fusion**, Zhiqiang Zhang et al. [[Paper](https://ieeexplore.ieee.org/document/9533925)] 
    * (arXiv preprint 2021) **CRD-CGAN: Category-Consistent and Relativistic Constraints for Diverse Text-to-Image Generation**, Tao Hu et al. [[Paper](https://arxiv.org/abs/2107.13516)]
    * (arXiv preprint 2021) **Improving Text-to-Image Synthesis Using Contrastive Learning**, Hui Ye et al. [[Paper](https://arxiv.org/pdf/2107.02423v1.pdf)] [[Code](https://github.com/huiyegit/T2I_CL)]
    * (arXiv preprint 2021) **CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders**, Kevin Frans et al. [[Paper](https://arxiv.org/pdf/2106.14843.pdf)] [[Code](https://colab.research.google.com/github/kvfrans/clipdraw/blob/main/clipdraw.ipynb)]
    * (ICASSP 2021) **Drawgan: Text to Image Synthesis with Drawing Generative Adversarial Networks**, Zhiqiang Zhang et al. [[Paper](https://ieeexplore.ieee.org/document/9414166)] 
    * (arXiv preprint 2021) **Text to Image Generation with Semantic-Spatial Aware GAN**, Kai Hu et al. [[Paper](https://arxiv.org/pdf/2104.00567.pdf)] [[Code](https://github.com/wtliao/text2image)]
    * (IJCNN 2021) **DTGAN: Dual Attention Generative Adversarial Networks for Text-to-Image Generation**, Zhenxing Zhang et al. [[Paper](https://ieeexplore.ieee.org/abstract/document/9533527)] 
    * (CVPR 2021) **TediGAN: Text-Guided Diverse Image Generation and Manipulation**, Weihao Xia et al. [[Paper](https://arxiv.org/pdf/2012.03308.pdf)] [[Extended Version](https://arxiv.org/pdf/2104.08910.pdf)][[Code](https://github.com/IIGROUP/TediGAN)] [[Dataset](https://github.com/IIGROUP/Multi-Modal-CelebA-HQ-Dataset)] [[Colab](https://colab.research.google.com/github/weihaox/TediGAN/blob/main/playground.ipynb)] [[Video](https://www.youtube.com/watch?v=L8Na2f5viAM)] 
    * (CVPR 2021) **Cross-Modal Contrastive Learning for Text-to-Image Generation**, Han Zhang et al. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Cross-Modal_Contrastive_Learning_for_Text-to-Image_Generation_CVPR_2021_paper.pdf)] [[Code](https://github.com/google-research/xmcgan_image_generation)]
    * (NeurIPS 2021) **CogView: Mastering Text-to-Image Generation via Transformers**, Ming Ding et al. [[Paper](https://arxiv.org/pdf/2105.13290.pdf)] [[Code](https://github.com/THUDM/CogView)] [[Demo Website(Chinese)](https://lab.aminer.cn/cogview/index.html)] 
    * (IEEE Transactions on Multimedia 2021) **Modality Disentangled Discriminator for Text-to-Image Synthesis**, Fangxiang Feng et al. [[Paper](https://ieeexplore.ieee.org/document/9417738)] [[Code](https://github.com/FangxiangFeng/DM-GAN-MDD)]
    * ⭐(arXiv preprint 2021) **Zero-Shot Text-to-Image Generation**, Aditya Ramesh et al. [[Paper](https://arxiv.org/pdf/2102.12092.pdf)] [[Code](https://github.com/openai/DALL-E)] [[Blog](https://openai.com/blog/dall-e/)] [[Model Card](https://github.com/openai/DALL-E/blob/master/model_card.md)] [[Colab](https://colab.research.google.com/github/openai/DALL-E/blob/master/notebooks/usage.ipynb)] [[Code(Pytorch)](https://github.com/lucidrains/DALLE-pytorch)]
    <!--https://colab.research.google.com/drive/1KA2w8bA9Q1HDiZf5Ow_VNOrTaWW4lXXG?usp=sharing -->
    * (Pattern Recognition 2021) **Unsupervised text-to-image synthesis**, Yanlong Dong et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320320303769)] 
    * (WACV 2021) **Faces a la Carte: Text-to-Face Generation via Attribute Disentanglement**, Tianren Wang et al. [[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Wang_Faces_a_la_Carte_Text-to-Face_Generation_via_Attribute_Disentanglement_WACV_2021_paper.pdf)] 
    * (WACV 2021) **Text-to-Image Generation Grounded by Fine-Grained User Attention**, Jing Yu Koh et al. [[Paper](https://arxiv.org/pdf/2011.03775.pdf)] 
    * (IEEE TIP 2021) **Multi-Sentence Auxiliary Adversarial Networks for Fine-Grained Text-to-Image Synthesis**, Yanhua Yang et al. [[Paper](https://ieeexplore.ieee.org/document/9345477)]
    * (IEEE Access 2021) **DGattGAN: Cooperative Up-Sampling Based Dual Generator Attentional GAN on Text-to-Image Synthesis**, Han Zhang et al. [[Paper](https://ieeexplore.ieee.org/abstract/document/9352788)]
    
* <span id="head-2020"> **2020**  </span> **[       «🎯Back To Top»       ](#)**
    * (WIREs Data Mining and Knowledge Discovery 2020) **A survey and taxonomy of adversarial neural networks for text-to-image synthesis**, Jorge Agnese et al. [[Paper](https://onlinelibrary.wiley.com/doi/epdf/10.1002/widm.1345)] 
    * (TPAMI 2020) **Semantic Object Accuracy for Generative Text-to-Image Synthesis**, Tobias Hinz et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9184960)] [[Code](https://github.com/tohinz/semantic-object-accuracy-for-generative-text-to-image-synthesis)]
    * (IEEE TIP 2020) **KT-GAN: Knowledge-Transfer Generative Adversarial Network for Text-to-Image Synthesis**, Hongchen Tan et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9210842)]
    * (ACM Trans 2020) **End-to-End Text-to-Image Synthesis with Spatial Constrains**, Min Wang et al. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3391709)]
    * (Neural Networks) **Image manipulation with natural language using Two-sided Attentive Conditional Generative Adversarial Network**, DaweiZhu et al. [[Paper](https://reader.elsevier.com/reader/sd/pii/S0893608020303257?token=A8183D548464C26BB62C5D498DC6FB3D7A83D0EDFDB9E4B1DFFE39A3B0F9A2075E26A4E4BB333F203FF50A63F4EE93CC)]
    * (IEEE Access 2020) **TiVGAN: Text to Image to Video Generation With Step-by-Step Evolutionary Generator**, Doyeon Kim et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9171240)]
    * (IEEE Access 2020) **Dualattn-GAN: Text to Image Synthesis With Dual Attentional Generative Adversarial Network**, Yali Cai et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8930532)]
    * (ICCL 2020) **VICTR: Visual Information Captured Text Representation for Text-to-Image Multimodal Tasks**, Soyeon Caren Han et al. [[Paper](https://arxiv.org/pdf/2010.03182v3.pdf)] [[Code](https://github.com/usydnlp/VICTR)]
    * (ECCV 2020) **CPGAN: Content-Parsing Generative Adversarial Networks for Text-to-Image Synthesis**, Jiadong Liang et al. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58548-8_29)] [[Code](https://github.com/dongdongdong666/CPGAN)]
    * (CVPR 2020) **RiFeGAN: Rich Feature Generation for Text-to-Image Synthesis From Prior Knowledge**, Jun Cheng et al. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_RiFeGAN_Rich_Feature_Generation_for_Text-to-Image_Synthesis_From_Prior_Knowledge_CVPR_2020_paper.pdf)] 
    * (CVPR 2020) **CookGAN: Causality based Text-to-Image Synthesis**, Bin Zhu et al. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_CookGAN_Causality_Based_Text-to-Image_Synthesis_CVPR_2020_paper.pdf)]
    * (CVPR 2020 - Workshop) **SegAttnGAN: Text to Image Generation with Segmentation Attention**, Yuchuan Gou et al. [[Paper](https://arxiv.org/pdf/2005.12444.pdf)]
    * (IVPR 2020) **PerceptionGAN: Real-world Image Construction from Provided Text through Perceptual Understanding**, Kanish Garg et al. [[Paper](https://arxiv.org/pdf/2007.00977.pdf)]
    * (COLING 2020) **Leveraging Visual Question Answering to Improve Text-to-Image Synthesis**, Stanislav Frolov et al. [[Paper](https://arxiv.org/pdf/2010.14953.pdf)] 
    * (IRCDL 2020) **Text-to-Image Synthesis Based on Machine Generated Captions**, Marco Menardi et al. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-39905-4_7)] 
    * (arXiv preprint 2020) **DF-GAN: Deep fusion generative adversarial networks for Text-to-Image synthesis**, Ming Tao et al. [[Paper](https://arxiv.org/pdf/2008.05865.pdf)] [[Code](https://github.com/tobran/DF-GAN)] 
    * (arXiv preprint 2020) **MPG: A Multi-ingredient Pizza Image Generator with Conditional StyleGANs**, Fangda Han et al. [[Paper](https://arxiv.org/pdf/2012.02821.pdf)] 
    
* <span id="head-2019"> **2019**  </span> **[       «🎯Back To Top»       ](#)**
    * (IEEE TCSVT 2019) **Bridge-GAN: Interpretable Representation Learning for Text-to-image Synthesis**, Mingkuan Yuan et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8902154)] [[Code](https://github.com/PKU-ICST-MIPL/Bridge-GAN_TCSVT2019)]
    * (AAAI 2019) **Perceptual Pyramid Adversarial Networks for Text-to-Image Synthesis**, Minfeng Zhu et al. [[Paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4844)]
    * (AAAI 2019) **Adversarial Learning of Semantic Relevance in Text to Image Synthesis**, Miriam Cha et al. [[Paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4553)]
    * (NeurIPS 2019) **Learn, Imagine and Create: Text-to-Image Generation from Prior Knowledge**, Tingting Qiao et al. [[Paper](https://papers.nips.cc/paper/8375-learn-imagine-and-create-text-to-image-generation-from-prior-knowledge.pdf)] [[Code](https://github.com/qiaott/LeicaGAN)]
    * (NeurIPS 2019) **Controllable Text-to-Image Generation**, Bowen Li et al. [[Paper](https://papers.nips.cc/paper/2019/file/1d72310edc006dadf2190caad5802983-Paper.pdf)] [[Code](https://github.com/mrlibw/ControlGAN)]
    * (CVPR 2019) **DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis**, Minfeng Zhu et al. [[Paper](https://arxiv.org/pdf/1904.01310.pdf)] [[Code](https://github.com/MinfengZhu/DM-GAN)]
    * (CVPR 2019) **Object-driven Text-to-Image Synthesis via Adversarial Training**, Wenbo Li et al. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Object-Driven_Text-To-Image_Synthesis_via_Adversarial_Training_CVPR_2019_paper.pdf)] [[Code](https://github.com/jamesli1618/Obj-GAN)]
    * (CVPR 2019) **MirrorGAN: Learning Text-to-image Generation by Redescription**, Tingting Qiao et al. [[Paper](https://arxiv.org/pdf/1903.05854.pdf)] [[Code](https://github.com/qiaott/MirrorGAN)]
    * (CVPR 2019) **Text2Scene: Generating Abstract Scenes from Textual Descriptions**, Fuwen Tan et al. [[Paper](https://arxiv.org/pdf/1809.01110.pdf)] [[Code](https://github.com/uvavision/Text2Scene)]
    * (CVPR 2019) **Semantics Disentangling for Text-to-Image Generation**, Guojun Yin et al. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yin_Semantics_Disentangling_for_Text-To-Image_Generation_CVPR_2019_paper.pdf)] [[Website](https://gjyin91.github.io/projects/sdgan.html)]
    * (CVPR 2019) **Text Guided Person Image Synthesis**, Xingran Zhou et al. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_Text_Guided_Person_Image_Synthesis_CVPR_2019_paper.pdf)]
    * (ICCV 2019) **Semantics-Enhanced Adversarial Nets for Text-to-Image Synthesis**, Hongchen Tan et al. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tan_Semantics-Enhanced_Adversarial_Nets_for_Text-to-Image_Synthesis_ICCV_2019_paper.pdf)] 
    * (ICCV 2019) **Dual Adversarial Inference for Text-to-Image Synthesis**, Qicheng Lao et al. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lao_Dual_Adversarial_Inference_for_Text-to-Image_Synthesis_ICCV_2019_paper.pdf)] 
    * (ICCV 2019) **Tell, Draw, and Repeat: Generating and Modifying Images Based on Continual Linguistic Instruction**, Alaaeldin El-Nouby et al. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/El-Nouby_Tell_Draw_and_Repeat_Generating_and_Modifying_Images_Based_on_ICCV_2019_paper.pdf)] [[Code](https://github.com/Maluuba/GeNeVA)]
    * (BMVC 2019) **MS-GAN: Text to Image Synthesis with Attention-Modulated Generators and Similarity-aware Discriminators**, Fengling Mao et al. [[Paper](http://www.jdl.link/doc/2011/20191223_19-BMVC-MS-GAN-Mao-small.pdf)] 
    * (arXiv preprint 2019) **FTGAN: A Fully-trained Generative Adversarial Networks for Text to Face Generation**, Xiang Chen et al. [[Paper](https://arxiv.org/abs/1904.05729)]
    * (arXiv preprint 2019) **GILT: Generating Images from Long Text**, Ori Bar El et al. [[Paper](https://arxiv.org/pdf/1901.02404.pdf)] [[Code](https://github.com/netanelyo/Recipe2ImageGAN)]
    
* <span id="head-2018"> **2018**  </span> **[       «🎯Back To Top»       ](#)**
    * (TPAMI 2018) **StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks**, Han Zhang et al. [[Paper](https://arxiv.org/pdf/1710.10916.pdf)] [[Code](https://github.com/hanzhanggit/StackGAN-v2)]
    * (BMVC 2018) **MC-GAN: Multi-conditional Generative Adversarial Network for Image Synthesis**, Hyojin Park et al. [[Paper](https://arxiv.org/pdf/1805.01123.pdf)] [[Code](https://github.com/HYOJINPARK/MC_GAN)]
    * (CVPR 2018) **AttnGAN: Fine-grained text to image generation with attentional generative adversarial networks**, Tao Xu et al. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf)] [[Code](https://github.com/taoxugit/AttnGAN)]
    * (CVPR 2018) **Photographic Text-to-Image Synthesis with a Hierarchically-nested Adversarial Network**, Zizhao Zhang et al. [[Paper](https://arxiv.org/pdf/1802.09178.pdf)] [[Code](https://github.com/ypxie/HDGan)]
    * (CVPR 2018) **Inferring Semantic Layout for Hierarchical Text-to-Image Synthesis**, Seunghoon Hong et al. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hong_Inferring_Semantic_Layout_CVPR_2018_paper.pdf)] 
    * (CVPR 2018) **Image Generation from Scene Graphs**, Justin Johnson et al. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0764.pdf)] [[Code](https://github.com/google/sg2im)]
    * (ICLR 2018 - Workshop) **ChatPainter: Improving Text to Image Generation using Dialogue**, Shikhar Sharma et al. [[Paper](https://arxiv.org/pdf/1802.08216.pdf)] 
    * (ACMMM 2018) **Text-to-image Synthesis via Symmetrical Distillation Networks**, Mingkuan Yuan et al. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3240508.3240559)]
    * (WACV 2018) **C4Synth: Cross-Caption Cycle-Consistent Text-to-Image Synthesis**, K. J. Joseph et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8658689)]
    * (arXiv preprint 2018) **Text to Image Synthesis Using Generative Adversarial Networks**, Cristian Bodnar. [[Paper](https://arxiv.org/pdf/1805.00676.pdf)] 
    * (arXiv preprint 2018) **Text-to-image-to-text translation using cycle consistent adversarial networks**, Satya Krishna Gorti et al. [[Paper](https://arxiv.org/pdf/1808.04538.pdf)] [[Code](https://github.com/CSC2548/text2image2textGAN)]
    
* <span id="head-2017"> **2017**  </span> **[       «🎯Back To Top»       ](#)**
    * (ICCV 2017) **StackGAN: Text to photo-realistic image synthesis with stacked generative adversarial networks**, Han Zhang et al. [[Paper](https://arxiv.org/pdf/1612.03242.pdf)] [[Code](https://github.com/hanzhanggit/StackGAN)]
    * (ICIP 2017) **I2T2I: Learning Text to Image Synthesis with Textual Data Augmentation**, Hao Dong et al. [[Paper](https://arxiv.org/pdf/1703.06676.pdf)] [[Code](https://github.com/zsdonghao/im2txt2im)]
    * (MLSP 2017) **Adversarial nets with perceptual losses for text-to-image synthesis**, Miriam Cha et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8168140)]

* <span id="head-2016"> **2016**  </span> **[       «🎯Back To Top»       ](#)**
    * (ICML 2016) **Generative Adversarial Text to Image Synthesis**, Scott Reed et al. [[Paper](http://proceedings.mlr.press/v48/reed16.pdf)] [[Code](https://github.com/reedscot/icml2016)]
    * (NeurIPS 2016) **Learning What and Where to Draw**, Scott Reed et al. [[Paper](https://arxiv.org/pdf/1610.02454.pdf)] [[Code](https://github.com/reedscot/nips2016)]


## <span id="head5"> *5. Other Related Works* </span>
   * <span id="head-MM"> **⭐Multimodality⭐** </span> **[       «🎯Back To Top»       ](#)**
       * (arXiv preprint 2021) **Multimodal Conditional Image Synthesis with Product-of-Experts GANs**, Xun Huang et al. [[Paper](https://arxiv.org/abs/2112.05130)]  [[Project](https://deepimagination.cc/PoE-GAN/)]
         * Text-to-Image, Segmentation-to-Image, *Text+Segmentation/Sketch/Image → Image*, *Sketch+Segmentation/Image → Image*, *Segmentation+Image → Image*
       * (arXiv preprint 2021) **L-Verse: Bidirectional Generation Between Image and Text**, Taehoon Kim et al. [[Paper](https://arxiv.org/abs/2111.11133)] [[Code](https://github.com/tgisaturday/L-Verse)] 
         * Text-To-Image, Image-To-Text, Image Reconstruction 
       * (arXiv preprint 2021) **NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion**, Chenfei Wu et al. [[Paper](https://arxiv.org/pdf/2111.12417.pdf)] [[Code](https://github.com/microsoft/NUWA)]
         * Text-To-Image, Sketch-to-Image, Image Completion, Text-Guided Image Manipulation, Text-to-Video, Video Prediction, Sketch-to-Video, Text-Guided Video Manipulation

   * <span id="head-L2S"> **Label-set → Semantic maps** </span> **[       «🎯Back To Top»       ](#)**
       * (ECCV 2020) **Controllable image synthesis via SegVAE**, Yen-Chi Cheng et al. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520154.pdf)] [[Code](https://github.com/yccyenchicheng/SegVAE)]

   * <span id="head-TI2I"> **Text+Image → Image** </span> **[       «🎯Back To Top»       ](#)**
       * (arXiv preprint 2022) [💬Image Style Transfer] **StyleCLIPDraw: Coupling Content and Style in Text-to-Drawing Translation**, Peter Schaldenbrand et al. [[Paper](https://arxiv.org/abs/2202.12362)] [[Dataset](https://www.kaggle.com/pittsburghskeet/drawings-with-style-evaluation-styleclipdraw)] [[Code](https://github.com/pschaldenbrand/StyleCLIPDraw)] [[Demo](https://replicate.com/pschaldenbrand/style-clip-draw)]
       * (arXiv preprint 2022) [💬Image Style Transfer] **Name Your Style: An Arbitrary Artist-aware Image Style Transfer**, Zhi-Song Liu et al. [[Paper](https://arxiv.org/abs/2202.13562)]
       * (arXiv preprint 2022) [💬3D Avatar Generation] **Text and Image Guided 3D Avatar Generation and Manipulation**, Zehranaz Canfes et al. [[Paper](https://arxiv.org/abs/2202.06079)] [[Project](https://catlab-team.github.io/latent3D/)]
       * (arXiv preprint 2022) [💬Image Inpainting] **NÜWA-LIP: Language Guided Image Inpainting with Defect-free VQGAN**, Minheng Ni et al. [[Paper](https://arxiv.org/abs/2202.05009)]
       * (arXiv preprint 2021) [💬Semantic Diffusion Guidance] **More Control for Free! Image Synthesis with Semantic Diffusion Guidance**, Xihui Liu et al. [[Paper](https://arxiv.org/abs/2112.05744)] [[Project](https://xh-liu.github.io/sdg/)] 
         * Text-To-Image, Image-To-Image, Text+Image → Image 
       * ⭐(arXiv preprint 2021) [💬Text+Image → Video] **Make It Move: Controllable Image-to-Video Generation with Text Descriptions**, Yaosi Hu et al. [[Paper](https://arxiv.org/abs/2112.02815)]
       * (arXiv preprint 2021) [💬Hairstyle Transfer] **HairCLIP: Design Your Hair by Text and Reference Image**, Tianyi Wei et al. [[Paper](https://arxiv.org/abs/2112.05142)] [[Code](https://github.com/wty-ustc/HairCLIP)] 
       * (arXiv preprint 2021) [💬NeRF] **CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields**, Can Wang et al. [[Paper](https://arxiv.org/abs/2112.05139)] [[Code](https://github.com/cassiePython/CLIPNeRF)] [[Project](https://cassiepython.github.io/clipnerf/)]
       * (arXiv preprint 2021) [💬NeRF] **Zero-Shot Text-Guided Object Generation with Dream Fields**, Ajay Jain et al. [[Paper](https://arxiv.org/abs/2112.01455)]  [[Project](https://ajayj.com/dreamfields)]
       * (arXiv preprint 2021) [💬Style Transfer] **CLIPstyler: Image Style Transfer with a Single Text Condition**, Gihyun Kwon et al. [[Paper](https://arxiv.org/abs/2112.00374)] [[Code](https://github.com/paper11667/CLIPstyler)] 
       * (NeurIPS 2021) **Instance-Conditioned GAN**, Arantxa Casanova et al. [[Paper](https://arxiv.org/abs/2109.05070)] [[Code](https://github.com/facebookresearch/ic_gan)]
       * (ICCVW 2021) **CIGLI: Conditional Image Generation from Language & Image**, Xiaopeng Lu et al. [[Paper](https://openaccess.thecvf.com/content/ICCV2021W/CLVL/papers/Lu_CIGLI_Conditional_Image_Generation_From_Language__Image_ICCVW_2021_paper.pdf)] [[Code](https://github.com/vincentlux/CIGLI?utm_source=catalyzex.com)]
       * (arXiv preprint 2021) **Blended Diffusion for Text-driven Editing of Natural Images**, Omri Avrahami et al. [[Paper](https://arxiv.org/abs/2111.14818)] [[Code](https://github.com/omriav/blended-diffusion)]
       * (arXiv preprint 2021) **StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery**, Or Patashnik et al. [[Paper](https://arxiv.org/pdf/2103.17249.pdf)] [[Code](https://github.com/openai/DALL-E)]
       * (arXiv preprint 2021) **Paint by Word**, David Bau et al. [[Paper](https://arxiv.org/pdf/2103.10951.pdf)] 
       * ⭐(arXiv preprint 2021) **Zero-Shot Text-to-Image Generation**, Aditya Ramesh et al. [[Paper](https://arxiv.org/pdf/2102.12092.pdf)] [[Code](https://github.com/openai/DALL-E)] [[Blog](https://openai.com/blog/dall-e/)] [[Model Card](https://github.com/openai/DALL-E/blob/master/model_card.md)] [[Colab](https://colab.research.google.com/drive/1KA2w8bA9Q1HDiZf5Ow_VNOrTaWW4lXXG?usp=sharing)] 
       * (NeurIPS 2020) **Lightweight Generative Adversarial Networks for Text-Guided Image Manipulation**, Bowen Li et al. [[Paper](https://arxiv.org/pdf/2010.12136.pdf)]
       * (CVPR 2020) **ManiGAN: Text-Guided Image Manipulation**, Bowen Li et al. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_ManiGAN_Text-Guided_Image_Manipulation_CVPR_2020_paper.pdf)] [[Code](https://github.com/mrlibw/ManiGAN)]
       * (ACMMM 2020) **Text-Guided Neural Image Inpainting**, Lisai Zhang et al. [[Paper](https://arxiv.org/pdf/2004.03212.pdf)] [[Code](https://github.com/idealwhite/TDANet)]
       * (ACMMM 2020) **Describe What to Change: A Text-guided Unsupervised Image-to-Image Translation Approach**, Yahui Liu et al. [[Paper](https://arxiv.org/pdf/2008.04200.pdf)]
       * (NeurIPS 2018) **Text-adaptive generative adversarial networks: Manipulating images with natural language**, Seonghyeon Nam et al. [[Paper](http://papers.nips.cc/paper/7290-text-adaptive-generative-adversarial-networks-manipulating-images-with-natural-language.pdf)] [[Code](https://github.com/woozzu/tagan)]

   * <span id="head-L2I"> **Layout → Image** </span> **[       «🎯Back To Top»       ](#)**
       * (CVPR 2021 [AI for Content Creation Workshop](http://visual.cs.brown.edu/workshops/aicc2021/)) **High-Resolution Complex Scene Synthesis with Transformers**, Manuel Jahn et al. [[Paper](https://arxiv.org/pdf/2105.06458.pdf)] 
       * (CVPR 2021) **Context-Aware Layout to Image Generation with Enhanced Object Appearance**, Sen He et al. [[Paper](https://arxiv.org/pdf/2103.11897.pdf)] [[Code](https://github.com/wtliao/layout2img)] 
       
   * <span id="head-S2I"> **Speech → Image** </span> **[       «🎯Back To Top»       ](#)**
       *  (IEEE/ACM Transactions on Audio, Speech and Language Processing 2021) **Generating Images From Spoken Descriptions**, Xinsheng Wang et al. [[Paper](https://dl.acm.org/doi/10.1109/TASLP.2021.3053391)] [[Code](https://github.com/xinshengwang/S2IGAN)]  [[Project](https://xinshengwang.github.io/project/s2igan/)]
       *  (INTERSPEECH 2020)**[Extent Version👆] S2IGAN: Speech-to-Image Generation via Adversarial Learning**, Xinsheng Wang et al. [[Paper](https://arxiv.org/abs/2005.06968)]
       * (IEEE Journal of Selected Topics in Signal Processing 2020) **Direct Speech-to-Image Translation**, Jiguo Li et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9067083)] [[Code](https://github.com/smallflyingpig/speech-to-image-translation-without-text)] [[Project](https://smallflyingpig.github.io/speech-to-image/main)]
       
   * <span id="head-T2VR"> **Text → Visual Retrieval** </span> **[       «🎯Back To Top»       ](#)**
       * (CVPR 2021) **T2VLAD: Global-Local Sequence Alignment for Text-Video Retrieval**, Xiaohan Wang et al. [[Paper](https://arxiv.org/pdf/2104.10054.pdf)] 
       * (CVPR 2021) **Thinking Fast and Slow: Efficient Text-to-Visual Retrieval with Transformers**, Antoine Miech et al. [[Paper](https://arxiv.org/pdf/2103.16553.pdf)] 
       * (IEEE Access 2019) **Query is GAN: Scene Retrieval With Attentional Text-to-Image Generative Adversarial Network**, RINTARO YANAGI et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8868179)]
 
   * <span id="head-T2V"> **Text → Video** </span> **[       «🎯Back To Top»       ](#)**
       * (arXiv preprint 2021) [❌Genertation Task] **Transcript to Video: Efficient Clip Sequencing from Texts**, Yu Xiong et al. [[Paper](https://arxiv.org/pdf/2107.11851.pdf)] [[Project](http://www.xiongyu.me/projects/transcript2video/)]
       * (arXiv preprint 2021) **GODIVA: Generating Open-DomaIn Videos from nAtural Descriptions**, Chenfei Wu et al. [[Paper](https://arxiv.org/pdf/2104.14806.pdf)] 
       * (arXiv preprint 2021) **Text2Video: Text-driven Talking-head Video Synthesis with Phonetic Dictionary**, Sibo Zhang et al. [[Paper](https://arxiv.org/pdf/2104.14631.pdf)] 
       * (IEEE Access 2020) **TiVGAN: Text to Image to Video Generation With Step-by-Step Evolutionary Generator**, DOYEON KIM et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9171240)] 
       * (IJCAI 2019) **Conditional GAN with Discriminative Filter Generation for Text-to-Video Synthesis**, Yogesh Balaji et al. [[Paper](https://www.ijcai.org/Proceedings/2019/0276.pdf)] 
       * (IJCAI 2019) **IRC-GAN: Introspective Recurrent Convolutional GAN for Text-to-video Generation**, Kangle Deng et al. [[Paper](https://www.ijcai.org/Proceedings/2019/0307.pdf)] 
       * (AAAI 2018) **Video Generation From Text**, Yitong Li et al. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/12233)] 
       * (ACMMM 2017) **To create what you tell: Generating videos from captions**, Yingwei Pan et al. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3123266.3127905)] 
## <span id="head6"> *Contact Me* </span>

* [Yutong ZHOU](https://github.com/Yutong-Zhou-cv) in [Interaction Laboratory, Ritsumeikan University.](https://github.com/Rits-Interaction-Laboratory) ლ(╹◡╹ლ) 

* If you have any question, please feel free to contact Yutong ZHOU (E-mail: <zhou@i.ci.ritsumei.ac.jp>).
