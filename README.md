# <p align=center>`awesome Text_to_Image papers`</p>

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

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
* - [ ] [4. Paper With Code](#head4)
  * - [ ] [Survey](#head-Survey)
  * - [ ] [2021](#head-2021)
  * - [ ] [2020](#head-2020)
  * - [x] [2019](#head-2019)
  * - [x] [2018](#head-2018)
  * - [x] [2017](#head-2017)
  * - [x] [2016](#head-2016)
  
* - [ ] [5. Other Related Works](#head5)
  * - [ ] [Label-set → Semantic maps](#head-L2S)
  * - [ ] [Text+Image → Image](#head-TI2I)
  * - [ ] [Layout → Image](#head-L2I)
  * - [ ] [Text → Visual Retrieval](#head-T2VR)
  * - [ ] [Text → Video](#head-T2V)

* [*Contact Me*](#head6)

 ## <span id="head1"> *1.Description* </span>

* In the last few decades, the fields of Computer Vision (CV) and Natural Language Processing (NLP) have been made several major technological breakthroughs in deep learning research. Recently, researchers appear interested in combining semantic information and visual information in these traditionally independent fields. 
A number of studies have been conducted on the text-to-image synthesis techniques that transfer input textual description (keywords or sentences) into realistic images.

* Papers, codes and datasets for the text-to-image task are available here.

 ## <span id="head2"> *2.Quantitative Evaluation Metrics* </span>

* <span id="head-IS"> Inception Score (IS) </span> [[Paper](https://arxiv.org/pdf/1606.03498.pdf)] [[Python Code (Pytorch)](https://github.com/sbarratt/inception-score-pytorch)] [[Python Code (Tensorflow)](https://github.com/taki0112/GAN_Metrics-Tensorflow)]

* <span id="head-FID"> Fréchet Inception Distance (FID) </span> [[Paper](https://papers.nips.cc/paper/7240-gans-trained-by-a-two-time-scale-update-rule-converge-to-a-local-nash-equilibrium.pdf)] [[Python Code (Pytorch)](https://github.com/mseitzer/pytorch-fid)] [[Python Code (Tensorflow)](https://github.com/taki0112/GAN_Metrics-Tensorflow)]

* <span id="head-R"> R-precision </span> [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf)]

* <span id="head-L2"> L<sub>2</sub> error </span> [[Paper](https://papers.nips.cc/paper/7290-text-adaptive-generative-adversarial-networks-manipulating-images-with-natural-language.pdf)]

## <span id="head3"> *3.Datasets* </span>

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
  * **Detailed information (Text Descriptions):**  ⇒ [[Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Reed_Learning_Deep_Representations_CVPR_2016_paper.pdf)] [[Website](https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view)]
    * Descriptions per image: 10 Captions
    
* <span id="head-COCO"> **MS-COCO** </span>

  COCO is a large-scale object detection, segmentation, and captioning dataset.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://arxiv.org/pdf/1405.0312.pdf)] [[Website](https://cocodataset.org/#overview)]
    * Number of images: 120k (**Training**: 80k. **Testing**: 40k.)
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

## <span id="head4"> *4.Paper With Code* </span>

* <span id="head-Survey"> **Survey**  </span>
    * (2021) **Adversarial Text-to-Image Synthesis: A Review**, Stanislav Frolov et al. [[Paper](https://arxiv.org/pdf/2101.09983.pdf)] 
    * (2019) **A Survey and Taxonomy of Adversarial Neural Networks for Text-to-Image Synthesis**, Jorge Agnese et al. [[Paper](https://arxiv.org/pdf/1910.09399.pdf)] 
    
* <span id="head-2021"> **2021**  </span>
    * (CVPR 2021) **TediGAN: Text-Guided Diverse Image Generation and Manipulation**, Weihao Xia et al. [[Paper](https://arxiv.org/pdf/2012.03308.pdf)]  [[Code](https://github.com/IIGROUP/TediGAN)] [[Dataset](https://github.com/IIGROUP/Multi-Modal-CelebA-HQ-Dataset)] [[Colab](https://colab.research.google.com/github/weihaox/TediGAN/blob/main/playground.ipynb)] [[Video](https://www.youtube.com/watch?v=L8Na2f5viAM)] 
    * ⭐(arXiv preprint 2021) **Zero-Shot Text-to-Image Generation**, Aditya Ramesh et al. [[Paper](https://arxiv.org/pdf/2102.12092.pdf)] [[Code](https://github.com/openai/DALL-E)] [[Blog](https://openai.com/blog/dall-e/)] [[Model Card](https://github.com/openai/DALL-E/blob/master/model_card.md)] [[Colab](https://colab.research.google.com/drive/1KA2w8bA9Q1HDiZf5Ow_VNOrTaWW4lXXG?usp=sharing)] [[Code(Pytorch)](https://github.com/lucidrains/DALLE-pytorch)]
    * (Pattern Recognition 2021) **Unsupervised text-to-image synthesis**, Yanlong Dong et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320320303769)] 
    * (WACV 2021) **Faces a la Carte: Text-to-Face Generation via Attribute Disentanglement**, Tianren Wang et al. [[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Wang_Faces_a_la_Carte_Text-to-Face_Generation_via_Attribute_Disentanglement_WACV_2021_paper.pdf)] 
    * (WACV 2021) **Text-to-Image Generation Grounded by Fine-Grained User Attention**, Jing Yu Koh et al. [[Paper](https://arxiv.org/pdf/2011.03775.pdf)] 
    * (arXiv preprint 2021) **Cross-Modal Contrastive Learning for Text-to-Image Generation**, Han Zhang et al. [[Paper](https://arxiv.org/pdf/2101.04702.pdf)] 
    
* <span id="head-2020"> **2020**  </span>
    * (WIREs Data Mining and Knowledge Discovery 2020) **A survey and taxonomy of adversarial neural networks for text-to-image synthesis**, Jorge Agnese et al. [[Paper](https://onlinelibrary.wiley.com/doi/epdf/10.1002/widm.1345)] 
    * (TPAMI 2020) **Semantic Object Accuracy for Generative Text-to-Image Synthesis**, Tobias Hinz et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9184960)] [[Code](https://github.com/tohinz/semantic-object-accuracy-for-generative-text-to-image-synthesis)]
    * (TIP 2020) **KT-GAN: Knowledge-Transfer Generative Adversarial Network for Text-to-Image Synthesis**, Hongchen Tan et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9210842)]
    * (ACM Trans 2020) **End-to-End Text-to-Image Synthesis with Spatial Constrains**, Min Wang et al. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3391709)]
    * (Neural Networks) **Image manipulation with natural language using Two-sided Attentive Conditional Generative Adversarial Network**, DaweiZhu et al. [[Paper](https://reader.elsevier.com/reader/sd/pii/S0893608020303257?token=A8183D548464C26BB62C5D498DC6FB3D7A83D0EDFDB9E4B1DFFE39A3B0F9A2075E26A4E4BB333F203FF50A63F4EE93CC)]
    * (IEEE Access 2020) **TiVGAN: Text to Image to Video Generation With Step-by-Step Evolutionary Generator**, Doyeon Kim et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9171240)]
    * (IEEE Access 2020) **Dualattn-GAN: Text to Image Synthesis With Dual Attentional Generative Adversarial Network**, Yali Cai et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8930532)]
    * (ECCV 2020) **CPGAN: Content-Parsing Generative Adversarial Networks for Text-to-Image Synthesis**, Jiadong Liang et al. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58548-8_29)] [[Code](https://github.com/dongdongdong666/CPGAN)]
    * (CVPR 2020) **RiFeGAN: Rich Feature Generation for Text-to-Image Synthesis From Prior Knowledge**, Jun Cheng et al. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_RiFeGAN_Rich_Feature_Generation_for_Text-to-Image_Synthesis_From_Prior_Knowledge_CVPR_2020_paper.pdf)] 
    * (CVPR 2020) **CookGAN: Causality based Text-to-Image Synthesis**, Bin Zhu et al. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_CookGAN_Causality_Based_Text-to-Image_Synthesis_CVPR_2020_paper.pdf)]
    * (CVPR 2020 - Workshop) **SegAttnGAN: Text to Image Generation with Segmentation Attention**, Yuchuan Gou et al. [[Paper](https://arxiv.org/pdf/2005.12444.pdf)]
    * (IVPR 2020) **PerceptionGAN: Real-world Image Construction from Provided Text through Perceptual Understanding**, Kanish Garg et al. [[Paper](https://arxiv.org/pdf/2007.00977.pdf)]
    * (COLING 2020) **Leveraging Visual Question Answering to Improve Text-to-Image Synthesis**, Stanislav Frolov et al. [[Paper](https://arxiv.org/pdf/2010.14953.pdf)] 
    * (IRCDL 2020) **Text-to-Image Synthesis Based on Machine Generated Captions**, Marco Menardi et al. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-39905-4_7)] 
    * (arXiv preprint 2020) **TIME: Text and Image Mutual-Translation Adversarial Networks**, Bingchen Liu et al. [[Paper](https://arxiv.org/pdf/2005.13192.pdf)] 
    * (arXiv preprint 2020) **DF-GAN: Deep fusion generative adversarial networks for Text-to-Image synthesis**, Ming Tao et al. [[Paper](https://arxiv.org/pdf/2008.05865.pdf)] [[Code](https://github.com/tobran/DF-GAN)] 
    * (arXiv preprint 2020) **MPG: A Multi-ingredient Pizza Image Generator with Conditional StyleGANs**, Fangda Han et al. [[Paper](https://arxiv.org/pdf/2012.02821.pdf)] 
    
* <span id="head-2019"> **2019**  </span>
    * (IEEE TCSVT 2019) **Bridge-GAN: Interpretable Representation Learning for Text-to-image Synthesis**, Mingkuan Yuan et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8902154)] [[Code](https://github.com/PKU-ICST-MIPL/Bridge-GAN_TCSVT2019)]
    * (AAAI 2019) **Perceptual Pyramid Adversarial Networks for Text-to-Image Synthesis**, Minfeng Zhu et al. [[Web](https://www.aaai.org/ojs/index.php/AAAI/article/view/4844)]
    * (AAAI 2019) **Adversarial Learning of Semantic Relevance in Text to Image Synthesis**, Miriam Cha et al. [[Web](https://www.aaai.org/ojs/index.php/AAAI/article/view/4553)]
    * (NIPS 2019) **Learn, Imagine and Create: Text-to-Image Generation from Prior Knowledge**, Tingting Qiao et al. [[Paper](https://papers.nips.cc/paper/8375-learn-imagine-and-create-text-to-image-generation-from-prior-knowledge.pdf)] [[Code](https://github.com/qiaott/LeicaGAN)]
    * (NIPS 2019) **Controllable Text-to-Image Generation**, Bowen Li et al. [[Paper](https://papers.nips.cc/paper/2019/file/1d72310edc006dadf2190caad5802983-Paper.pdf)] [[Code](https://github.com/mrlibw/ControlGAN)]
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
    * (arXiv preprint 2019) **GILT: Generating Images from Long Text**, Ori Bar El et al. [[Paper](https://arxiv.org/pdf/1901.02404.pdf)] [[Code](https://github.com/netanelyo/Recipe2ImageGAN)]
    
* <span id="head-2018"> **2018**  </span>
    * (TPAMI 2018) **StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks**, Han Zhang et al. [[Paper](https://arxiv.org/pdf/1710.10916.pdf)] [[Code](https://github.com/hanzhanggit/StackGAN-v2)]
    * (BMVC 2018) **MC-GAN: Multi-conditional Generative Adversarial Network for Image Synthesis**, Hyojin Park et al. [[Paper](https://arxiv.org/pdf/1805.01123.pdf)] [[Code](https://github.com/HYOJINPARK/MC_GAN)]
    * (CVPR 2018) **AttnGAN: Fine-grained text to image generation with attentional generative adversarial networks**, Tao Xu et al. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf)] [[Code](https://github.com/taoxugit/AttnGAN)]
    * (CVPR 2018) **Photographic Text-to-Image Synthesis with a Hierarchically-nested Adversarial Network**, Zizhao Zhang et al. [[Paper](https://arxiv.org/pdf/1802.09178.pdf)] [[Code](https://github.com/ypxie/HDGan)]
    * (CVPR 2018) **Inferring Semantic Layout for Hierarchical Text-to-Image Synthesis**, Seunghoon Hong et al. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hong_Inferring_Semantic_Layout_CVPR_2018_paper.pdf)] 
    * (CVPR 2018) **Image Generation from Scene Graphs**, Justin Johnson et al. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0764.pdf)] [[Code](https://github.com/google/sg2im)]
    * (NIPS 2018) **Text-adaptive generative adversarial networks: Manipulating images with natural language**, Seonghyeon Nam et al. [[Paper](http://papers.nips.cc/paper/7290-text-adaptive-generative-adversarial-networks-manipulating-images-with-natural-language.pdf)] [[Code](https://github.com/woozzu/tagan)]
    * (ICLR 2018 - Workshop) **ChatPainter: Improving Text to Image Generation using Dialogue**, Shikhar Sharma et al. [[Paper](https://arxiv.org/pdf/1802.08216.pdf)] 
    * (ACMMM 2018) **Text-to-image Synthesis via Symmetrical Distillation Networks**, Mingkuan Yuan et al. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3240508.3240559)]
    * (WACV 2018) **C4Synth: Cross-Caption Cycle-Consistent Text-to-Image Synthesis**, K. J. Joseph et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8658689)]
    * (arXiv preprint 2018) **Text to Image Synthesis Using Generative Adversarial Networks**, Cristian Bodnar. [[Paper](https://arxiv.org/pdf/1805.00676.pdf)] 
    * (arXiv preprint 2018) **Text-to-image-to-text translation using cycle consistent adversarial networks**, Satya Krishna Gorti et al. [[Paper](https://arxiv.org/pdf/1808.04538.pdf)] [[Code](https://github.com/CSC2548/text2image2textGAN)]
    
* <span id="head-2017"> **2017**  </span>
    * (ICCV 2017) **StackGAN: Text to photo-realistic image synthesis with stacked generative adversarial networks**, Han Zhang et al. [[Paper](https://arxiv.org/pdf/1612.03242.pdf)] [[Code](https://github.com/hanzhanggit/StackGAN)]
    * (ICIP 2017) **I2T2I: Learning Text to Image Synthesis with Textual Data Augmentation**, Hao Dong et al. [[Paper](https://arxiv.org/pdf/1703.06676.pdf)] [[Code](https://github.com/zsdonghao/im2txt2im)]
    * (MLSP 2017) **Adversarial nets with perceptual losses for text-to-image synthesis**, Miriam Cha et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8168140)]

* <span id="head-2016"> **2016**  </span>
    * (ICML 2016) **Generative Adversarial Text to Image Synthesis**, Scott Reed et al. [[Paper](http://proceedings.mlr.press/v48/reed16.pdf)] [[Code](https://github.com/reedscot/icml2016)]
    * (NIPS 2016) **Learning What and Where to Draw**, Scott Reed et al. [[Paper](https://arxiv.org/pdf/1610.02454.pdf)] [[Code](https://github.com/reedscot/nips2016)]


## <span id="head5"> *5. Other Related Works* </span>
   * <span id="head-L2S"> **Label-set → Semantic maps** </span>
       * (ECCV 2020) **Controllable image synthesis via SegVAE**, Yen-Chi Cheng et al. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520154.pdf)] [[Code](https://github.com/orpatashnik/StyleCLIP)]

   * <span id="head-TI2I"> **Text+Image → Image** </span>
       * (arXiv preprint 2021) **StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery**, Or Patashnik et al. [[Paper](https://arxiv.org/pdf/2103.17249.pdf)] [[Code](https://github.com/openai/DALL-E)]
       * (arXiv preprint 2021) **Paint by Word**, David Bau et al. [[Paper](https://arxiv.org/pdf/2103.10951.pdf)] 
       * ⭐(arXiv preprint 2021) **Zero-Shot Text-to-Image Generation**, Aditya Ramesh et al. [[Paper](https://arxiv.org/pdf/2102.12092.pdf)] [[Code](https://github.com/openai/DALL-E)] [[Blog](https://openai.com/blog/dall-e/)] [[Model Card](https://github.com/openai/DALL-E/blob/master/model_card.md)] [[Colab](https://colab.research.google.com/drive/1KA2w8bA9Q1HDiZf5Ow_VNOrTaWW4lXXG?usp=sharing)] 
       * (NIPS 2020) **Lightweight Generative Adversarial Networks for Text-Guided Image Manipulation**, Bowen Li et al. [[Paper](https://arxiv.org/pdf/2010.12136.pdf)]
       * (CVPR 2020) **ManiGAN: Text-Guided Image Manipulation**, Bowen Li et al. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_ManiGAN_Text-Guided_Image_Manipulation_CVPR_2020_paper.pdf)] [[Code](https://github.com/mrlibw/ManiGAN)]
       * (ACMMM 2020) **Text-Guided Neural Image Inpainting**, Lisai Zhang et al. [[Paper](https://arxiv.org/pdf/2004.03212.pdf)] [[Code](https://github.com/idealwhite/TDANet)]
       * (ACMMM 2020) **Describe What to Change: A Text-guided Unsupervised Image-to-Image Translation Approach**, Yahui Liu et al. [[Paper](https://arxiv.org/pdf/2008.04200.pdf)]

   * <span id="head-L2I"> **Layout → Image** </span>
       * (CVPR 2021) **Context-Aware Layout to Image Generation with Enhanced Object Appearance**, Sen He et al. [[Paper](https://arxiv.org/pdf/2103.11897.pdf)] [[Code](https://github.com/wtliao/layout2img)]

   * <span id="head-T2VR"> **Text → Visual Retrieval** </span>
       * (CVPR 2021) **T2VLAD: Global-Local Sequence Alignment for Text-Video Retrieval**, Xiaohan Wang et al. [[Paper](https://arxiv.org/pdf/2104.10054.pdf)] 
       * (CVPR 2021) **Thinking Fast and Slow: Efficient Text-to-Visual Retrieval with Transformers**, Antoine Miech et al. [[Paper](https://arxiv.org/pdf/2103.16553.pdf)] 
 
   * <span id="head-T2V"> **Text → Video** </span>
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
