# <p align=center>𝓐𝔀𝓮𝓼𝓸𝓶𝓮 𝓣𝓮𝔁𝓽📝-𝓽𝓸-𝓘𝓶𝓪𝓰𝓮🌇</p>
<!--# <p align=center>`Awesome Text📝-to-Image🌇`</p>-->
<div align=center>

<p>
 
 ![GitHub stars](https://img.shields.io/github/stars/Yutong-Zhou-cv/Awesome-Text-to-Image.svg?color=red&style=for-the-badge) 
 ![GitHub forks](https://img.shields.io/github/forks/Yutong-Zhou-cv/Awesome-Text-to-Image.svg?style=for-the-badge) 
 ![GitHub activity](https://img.shields.io/github/last-commit/Yutong-Zhou-cv/Awesome-Text-to-Image?color=yellow&style=for-the-badge) 
 ![GitHub issues](https://img.shields.io/github/issues/Yutong-Zhou-cv/Awesome-Text-to-Image?style=for-the-badge)
 ![GitHub closed issues](https://img.shields.io/github/issues-closed/Yutong-Zhou-cv/Awesome-Text-to-Image?color=inactive&style=for-the-badge)
 
 [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) 
 ![Visitors](https://visitor-badge.glitch.me/badge?page_id=Yutong-Zhou-cv/Awesome-Text-to-Image) 
 [![Star History Chart](https://api.star-history.com/svg?repos=Yutong-Zhou-cv/Awesome-Text-to-Image&type=Date)](https://star-history.com/#Yutong-Zhou-cv/Awesome-Text-to-Image&Date)
</p>

𝓐 𝓬𝓸𝓵𝓵𝓮𝓬𝓽𝓲𝓸𝓷 𝓸𝓯 𝓻𝓮𝓼𝓸𝓾𝓻𝓬𝓮𝓼 𝓸𝓷 𝓽𝓮𝔁𝓽-𝓽𝓸-𝓲𝓶𝓪𝓰𝓮 𝓼𝔂𝓷𝓽𝓱𝓮𝓼𝓲𝓼/𝓶𝓪𝓷𝓲𝓹𝓾𝓵𝓪𝓽𝓲𝓸𝓷 𝓽𝓪𝓼𝓴𝓼.
 
</div>

![Figure from paper](pic/Overview.png)
> *From: [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://cdn.openai.com/papers/dall-e-2.pdf)*

## *<img src="./pic/icon/to-do-list.png" width="25" /> To Do*
* - [ ] Add [**Best Collection**]() for Awesome-Text-to-Image
* - [ ] Add **Topic Order** list and **Chronological Order** list

## <span id="head-content"> *Content* </span>
* - [x] [1. Description](#head1)

* - [x] [2. Quantitative Evaluation Metrics](#head2)
  * [Inception Score (IS)](#head-IS)
  * [Fréchet Inception Distance (FID)](#head-FID)  
  * [R-precision](#head-R)
  * [L<sub>2</sub> error](#head-L2)
  * [Learned Perceptual Image Patch Similarity (LPIPS)](#head-LPIPS)
  
* - [ ] [3. Datasets](#head3)  
  * [Caltech-UCSD Bird (CUB)](#head-CUB)
  * [Oxford-102 Flower](#head-Flower)
  * [MS-COCO](#head-COCO)
  * [Multi-Modal-CelebA-HQ](#head-Multi-Modal-CelebA-HQ)
  * [CelebA-Dialog](#head-CelebA-Dialog)
  * [FFHQ-Text](#head-FFHQ-Text)
  * [CelebAText-HQ](#head-CelebAText-HQ)
  * [DeepFashion-MultiModal](#head-DeepFashion-MultiModal)
  * [ANNA](#head-ANNA)

* - [ ] [4. Project](#head4)

* - [ ] [5. ⏳Recently Focused Papers (FYI)](#head5)

* - [ ] [6. Paper With Code](#head6)
  * - [ ] [Survey](#head-Survey)
  * - [ ] [Text to Face👨🏻🧒👧🏼🧓🏽](#head-T2F)
  * - [ ] [Compounding Issues🤔](#head-CI)
  * - [ ] [2023](#head-2023)
  * - [ ] [2022](#head-2022)
  * - [x] [2021](#head-2021)
  * - [x] [2020](#head-2020)
  * - [x] [2019](#head-2019)
  * - [x] [2018](#head-2018)
  * - [x] [2017](#head-2017)
  * - [x] [2016](#head-2016)
  
* - [ ] [7. Other Related Works](#head7)
  * - [ ] [⭐Multimodality⭐](#head-MM)
  * - [ ] [🛫Applications🛫](#head-APP)
  * - [ ] [Text+Image/Video → Image/Video](#head-TI2I)
  * - [ ] [Audio+Text+Image/Video → Image/Video](#head-ATI2I)
  * - [ ] [Layout → Image](#head-L2I)
  * - [ ] [Label-set → Semantic maps](#head-L2S)
  * - [ ] [Speech → Image](#head-S2I)
  * - [ ] [Text → Visual Retrieval](#head-T2VR)
  * - [ ] [Text → Motion/Shape/Mesh/Object...](#head-T2M)
  * - [ ] [Text → Video](#head-T2V)
  * - [ ] [Text → Music](#head-T2Mu)

* [*Contact Me*](#head6)

 ## <span id="head1"> *1. Description* </span>

* In the last few decades, the fields of Computer Vision (CV) and Natural Language Processing (NLP) have been made several major technological breakthroughs in deep learning research. Recently, researchers appear interested in combining semantic information and visual information in these traditionally independent fields. 
A number of studies have been conducted on the text-to-image synthesis techniques that transfer input textual description (keywords or sentences) into realistic images.

* Papers, codes and datasets for the text-to-image task are available here.

>🐌 Markdown Format:
> * (Conference/Journal Year) **Title**, First Author et al. [[Paper](URL)] [[Code](URL)] [[Project](URL)]

 ## <span id="head2"> *2. Quantitative Evaluation Metrics* </span> [       «🎯Back To Top»       ](#)

* <span id="head-IS"> Inception Score (IS) </span> [[Paper](https://arxiv.org/pdf/1606.03498.pdf)] [[Python Code (Pytorch)](https://github.com/sbarratt/inception-score-pytorch)] [(New!)[Python Code (Tensorflow)](https://github.com/senmaoy/Inception-Score-FID-on-CUB-and-OXford)] [[Python Code (Tensorflow)](https://github.com/taki0112/GAN_Metrics-Tensorflow)] [[Ref.Code(AttnGAN)](https://github.com/taoxugit/AttnGAN)]

* <span id="head-FID"> Fréchet Inception Distance (FID) </span> [[Paper](https://papers.nips.cc/paper/7240-gans-trained-by-a-two-time-scale-update-rule-converge-to-a-local-nash-equilibrium.pdf)] [[Python Code (Pytorch)](https://github.com/mseitzer/pytorch-fid)] [(New!)[Python Code (Tensorflow)](https://github.com/senmaoy/Inception-Score-FID-on-CUB-and-OXford)] [[Python Code (Tensorflow)](https://github.com/taki0112/GAN_Metrics-Tensorflow)] [[Ref.Code(DM-GAN)](https://github.com/MinfengZhu/DM-GAN)]

* <span id="head-R"> R-precision </span> [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf)] [[Ref.Code(CPGAN)](https://github.com/dongdongdong666/CPGAN)]

* <span id="head-L2"> L<sub>2</sub> error </span> [[Paper](https://papers.nips.cc/paper/7290-text-adaptive-generative-adversarial-networks-manipulating-images-with-natural-language.pdf)]

* <span id="head-LPIPS"> Learned Perceptual Image Patch Similarity (LPIPS) </span> [[Paper](https://arxiv.org/abs/1801.03924)] [[Python Code](https://github.com/richzhang/PerceptualSimilarity)]

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

* <span id="head-CelebA-Dialog"> **CelebA-Dialog** </span>

  CelebA-Dialog is a large-scale visual-language face dataset. It has two properties:
(1) Facial images are annotated with **rich fine-grained labels**, which classify one attribute into multiple degrees according to its semantic meaning.
(2) Accompanied with each image, there are **captions describing** the attributes and a **user request sample**.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Talk-To-Edit_Fine-Grained_Facial_Editing_via_Dialog_ICCV_2021_paper.pdf)] [[Website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Dialog.html)] [[Download](https://github.com/yumingj/Talk-to-Edit)]
    * Number of identities: 10,177
    * Number of images: 202,599 
    * 5 fine-grained attributes annotations per image: Bangs, Eyeglasses, Beard, Smiling, and Age


* <span id="head-FFHQ-Text"> **FFHQ-Text** </span>

  FFHQ-Text is a small-scale face image dataset with large-scale facial attributes, designed for text-to-face generation & manipulation, text-guided facial image manipulation, and other vision-related tasks.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3481026)] [[Website](https://github.com/Yutong-Zhou-cv/FFHQ-Text_Dataset)] [[Download](https://forms.gle/f7oMXD3g9BgdgEUd7)]
    * Number of images (from FFHQ): 760 (**Training**: 500. **Testing**: 260.)
    * Descriptions per image: 9 Captions
    * 13 multi-valued facial element groups from coarse to fine.
  * **Detailed information (BBox):** ⇒ [[Website](https://www.robots.ox.ac.uk/~vgg/software/via/)]

* <span id="head-CelebAText-HQ"> **CelebAText-HQ** </span>

  CelebAText-HQ is a large-scale face image dataset with large-scale facial attributes, designed for text-to-face generation.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475391)] [[Website](https://github.com/cripac-sjx/SEA-T2F)] [[Download](https://drive.google.com/drive/folders/1IAb_iy6-soEGQWhbgu6cQODsIUJZpahC)]
    * Number of images (from Celeba-HQ): 15010 (**Training**: 13,710. **Testing**: 1300.)
    * Descriptions per image: 10 Captions

* <span id="head-DeepFashion-MultiModal"> **DeepFashion-MultiModal** </span>
  
  CelebA-Dialog is a large-scale high-quality human dataset. Human images are annotated with **rich multi-modal labels**, including human parsing labels, keypoints, densepose, fine-grained attributes and textual descriptions.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://arxiv.org/pdf/2205.15996.pdf)] [[Website](https://github.com/yumingj/DeepFashion-MultiModal)] [[Download](https://drive.google.com/drive/folders/1An2c_ZCkeGmhJg0zUjtZF46vyJgQwIr2?usp=sharing)]
    * Number of images: 44,096, including 12,701 full body images
    * Descriptions per image: 1 Caption

* <span id="head-ANNA"> **ANNA** </span>

  ANNA is an Abstractive News captioNs dAtaset extracted from online news articles in a variety of different contexts. The generated images are judged on the basis of contextual relevance, visual quality, and perceptual similarity to ground-truth image-caption pairs.
  * **Detailed information (Images & Text Descriptions):**  ⇒ [[Paper](https://arxiv.org/abs/2301.02160)] [[Download](https://github.com/aashish2000/ANNA)]
    * Number of image-text pairs (from The New York Times): 29625 (**Training**: 17897. **Validation**: 1622. **Testing**: 1649.)


## <span id="head4"> *4. Project* </span> **[       «🎯Back To Top»       ](#)**
* **2022 Top Papers in AI — A Year of Generative Models**. [[Medium](https://chuanenlin.medium.com/2022-top-ai-papers-a-year-of-generative-models-a7dcd9109e39)] 
* **Whisper-to-stable-diffusion**. [[Hugging Face](https://huggingface.co/spaces/fffiloni/whisper-to-stable-diffusion)] 
    * >***Whisper*** is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio. It is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.
    * >***Stable Diffusion*** is a state-of-the-art text-to-image model that generates images from text.
* ⭐⭐**Stable Diffusion 2.0**. [[Github](https://github.com/Stability-AI/stablediffusion)] [[Website](https://stability.ai/blog/stable-diffusion-v2-release)] [[Stability AI API Platform](https://platform.stability.ai/)] [[DreamStudio Beta](https://beta.dreamstudio.ai/dream)]
    * >***New Text-to-Image Diffusion Model:*** Generates images with default resolutions of both 512x512 pixels and 768x768 pixels and remove adult content using LAION’s NSFW filter.
    * >***Super-resolution Upscaler Diffusion Model:*** Enhances the resolution of images by a factor of 4.
    * >***Depth-to-Image Diffusion Model:*** Depth2img infers the depth of an input image (using [an existing model](https://github.com/isl-org/MiDaS)), and then generates new images using both the text and depth information. New images look radically different from the original but still preserve the coherence and depth of that image.
    * >***Updated Inpainting Diffusion Model:*** A new text-guided inpainting model, fine-tuned on the new Stable Diffusion 2.0 base text-to-image, which makes it super easy to switch out parts of an image intelligently and quickly.
 
* ⭐**Stable Diffusion**. [[**Awesome Stable-Diffusion**](https://github.com/awesome-stable-diffusion/awesome-stable-diffusion)] [[Researcher Access Form](https://stability.ai/research-access-form)] [[Github](https://github.com/CompVis/stable-diffusion)] [[Web UI](https://github.com/sd-webui/stable-diffusion-webui)] [[WebUI Docker](https://github.com/AbdBarho/stable-diffusion-webui-docker)] [[Hugging Face](https://huggingface.co/spaces/stabilityai/stable-diffusion)] [[DreamStudio Beta](https://beta.dreamstudio.ai/dream)]
    * >Stable Diffusion is a text-to-image model that will empower billions of people to create stunning art within seconds. It is a breakthrough in speed and quality meaning that it can run on consumer GPUs.
* **Midjourney**. [[Documentation](https://midjourney.gitbook.io/docs/)] [[Homepage](https://www.midjourney.com/home/)] 
    * >Midjourney is an independent research lab exploring new mediums of thought and expanding the imaginative powers of the human species.
There are two ways to experience the tools: the Midjourney Bot and the [Web App](https://www.midjourney.com/).
* **Artflowq**. [[Start Creating!](https://artflow.ai/)] 
    * >Artflow lets users generate visual content with the help of AI. Create unique avatars with ease and turn any description into a portrait.
* **craiyon(~~DALL·E Mini~~)**. [[Short Video Explanation](https://www.youtube.com/watch?v=qOxde_JV0vI)] [[Blog](https://www.louisbouchard.ai/dalle-mini/)] [[Github](https://github.com/borisdayma/dalle-mini)] [[Huggingface official demo](https://huggingface.co/spaces/dalle-mini/dalle-mini)] [[Homepage](https://www.craiyon.com/)] [[min(DALL·E)](https://github.com/kuprel/min-dalle)]
    * >A free, open-source AI that produces amazing images from text inputs. AI model drawing images from any prompt!
* **Disco Diffusion**. [[Github](https://github.com/alembics/disco-diffusion)] [[Colab](https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb)]
    * >A frankensteinian amalgamation of notebooks, models and techniques for the generation of AI Art and Animations.
* **Aphantasia**. [[Github](https://github.com/eps696/aphantasia)] 
    * >This is a text-to-image tool, part of the [artwork](https://computervisionart.com/pieces2021/aphantasia/) of the same name. ([Aphantasia](https://en.wikipedia.org/wiki/Aphantasia) is the inability to visualize mental images, the deprivation of visual dreams.)
* **Text2Art**. [[Try it now!](https://text2art.com/)] [[Github](https://github.com/mfrashad/text2art)] [[Blog](https://towardsdatascience.com/how-i-built-an-ai-text-to-art-generator-a0c0f6d6f59f)] 
    * >Text2Art is an AI-powered art generator based on VQGAN+CLIP that can generate all kinds of art such as pixel art, drawing, and painting from just text input. 
* **Survey Text Based Image Synthesis** [[Blog](https://hackmd.io/@prajwalsingh/imagesynthesis#) (2021)]

## <span id="head5"> *5. ⏳Recently Focused Papers* (FYI) </span> **[       «🎯Back To Top»       ](#)**
* ⭐(arXiv preprint 2022) *Text-To-4D Dynamic Scene Generation*, Uriel Singer et al. [[Paper](https://arxiv.org/abs/2301.11280)] [[Project](https://make-a-video3d.github.io/)]
    * 🍬 MAV3D: ***the first*** generate 3D dynamic scenes given a text description. MAV3D ***does not require any 3D or 4D data*** and the T2V model is trained only on ***Text-Image pairs and unlabeled videos***. 
* ⭐⭐(arXiv preprint 2023) *Muse: Text-To-Image Generation via Masked Generative Transformers*, Huiwen Chang et al. [[Paper](https://arxiv.org/abs/2301.00704v1)] [[Project](https://muse-model.github.io/)] 
    * 🍬 Muse: a ***state-of-the-art*** model for text-to-image generation which achieves excellent FID and CLIP scores; significantly ***faster*** than comparable models; enables ***out-of-the-box, zero-shot editing capabilities*** including inpainting, outpainting, and mask-free editing.
* ⭐(arXiv preprint 2022) *ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts*, Zhida Feng et al.  [[Paper](https://arxiv.org/abs/2210.15257)] 
    * 🍬 ERNIE-ViLG 2.0: a large-scale ***Chinese text-to-image diffusion model***, which progressively upgrades the quality of generated images by: (1) incorporating fine-grained textual and visual knowledge of key elements in the scene, and (2) utilizing different denoising experts at different denoising stages. ERNIE-ViLG 2.0 ***achieves state-of-the-art on MS-COCO*** with a ***zero-shot FID score of 6.75***.
* ⭐⭐(arXiv preprint 2022) *Prompt-to-Prompt Image Editing with Cross Attention Control*, Amir Hertz et al.  [[Paper](https://arxiv.org/abs/2208.01626)] [[Code](https://github.com/google/prompt-to-prompt)] [[Unofficial Code](https://github.com/bloc97/CrossAttentionControl)] [[Project](https://prompt-to-prompt.github.io/)]
    * 🍬 Prompt-to-Prompt Editing: ***Control the attention maps*** of the edited image by injecting the attention maps of the original image along the diffusion process. Monitor the synthesis process by ***editing the textual prompt only***, paving the way to a myriad of caption-based editing applications.

<!--
### <p align=center> **🍃Update History**</p>
*(2023.01.12 Update)*
* ⭐⭐(arXiv preprint 2022) *Imagen Video: High Definition Video Generation with Diffusion Models*, Jonathan Ho et al. [[Paper](https://arxiv.org/abs/2210.02303v1)] [[Project](https://imagen.research.google/video/)] 
    * 🍬 Imagen Video: Given a text prompt, Imagen Video generates high-definition videos using a base video generation model and a sequence of interleaved spatial and temporal video super-resolution models. Imagen Video is not only capable of ***generating videos of high fidelity***, but also ***having a high degree of controllability and world knowledge***, including the ability to generate diverse videos and text animations in various artistic styles and with 3D object understanding. 
* ⭐⭐(arXiv preprint 2022) *Make-A-Video: Text-to-Video Generation without Text-Video Data*, Uriel Singer et al. [[Paper](https://arxiv.org/abs/2209.14792)] [[Project](https://makeavideo.studio/)] [[Short read](https://www.louisbouchard.ai/make-a-video/)] [[Code](https://github.com/lucidrains/make-a-video-pytorch)]
    * 🍬 Meta AI’s new model make-a-video is out and in a single sentence: ***it generates videos from text***. It’s not only able to generate videos, but it’s also ***the new state-of-the-art method***, producing ***higher quality and more coherent videos*** than ever before!
* ⭐⭐(arXiv preprint 2022) *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*, Nataniel Ruiz et al. [[Paper](https://arxiv.org/abs/2208.12242)] [[Project](https://dreambooth.github.io/)]
    * 🍬 DreamBooth: Given as ***input just a few images*** of a subject and fine-tune a pretrained text-to-image model (Imagen), such that it learns to ***bind a unique identifier with that specific subject***, which synthesizing the subject in ***diverse scenes, poses, views, and lighting conditions*** that do not appear in the reference images. 
    * 📚 Subject Recontextualization, Text-guided View Synthesis, Appearance Modification, Artistic Rendering (all while preserving the subject's key features)
*(2022.10.28 Update)*
* ⭐(arXiv preprint 2022) *An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion*, Rinon Gal et al.  [[Paper](https://arxiv.org/abs/2208.01618)] [[Code](https://github.com/rinongal/textual_inversion)] [[Project](https://textual-inversion.github.io/)]
    * Using ***only 3-5 images*** of a user-provided concept (an object or a style) learn to represent it through ***new "words" in the embedding space*** of a frozen text-to-image model. These "words" can be composed into natural language sentences, ***guiding personalized creation in an intuitive way***. 
* ⭐⭐(ECCV 2022) *Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors*, Oran Gafni et al. [[Paper](https://arxiv.org/abs/2203.13131)] [[Code](https://github.com/CasualGANPapers/Make-A-Scene)] [[(Story)The Little Red Boat Story](https://www.youtube.com/watch?v=N4BagnXzPXY)] [[(Story)New Adventures](https://www.youtube.com/watch?v=NPTFNNACRnI&t=154s)]
    * 🍬 Make-A-Scene: Generate high fidelity images in a resolution of ***512x512*** pixels; Introduce several new capabilities: (i) ***Scene editing***, (ii) ***text editing with anchor scenes***, (iii) ***overcoming out-of-distribution text prompts***, and (iv) ***story illustration generation***.

*(2022.09.30 Update)*
* ⭐(arXiv preprint 2022) *NUWA-Infinity: Autoregressive over Autoregressive Generation for Infinite Visual Synthesis*, Chenfei Wu et al. [[Paper](https://arxiv.org/abs/2207.09814)] [[Code](https://github.com/microsoft/NUWA)] [[Project](https://nuwa-infinity.microsoft.com/#/)]
    * 🍬 NUWA-Infinity: A infinite visual synthesis model for ***arbitrarily-sized high-resolution images*** and ***long-duration videos*** generation.
* ⭐⭐(arXiv preprint 2022) *Scaling Autoregressive Models for Content-Rich Text-to-Image Generation*, Jiahui Yu et al. [[Paper](https://arxiv.org/abs/2206.10789)] [[Code](https://github.com/google-research/parti)] [[Project](https://parti.research.google/)]
    * 🍬 Pathways Autoregressive Text-to-Image (Parti): Generate high-fidelity photorealistic images and supports ***content-rich synthesis involving complex compositions and world knowledge***; Treats text-to-image generation as a sequence-to-sequence modeling problem, ***akin to machine translation***, with sequences of image tokens as the target outputs rather than text tokens in another language. 

*(2022.08.28 Update)*
* ⭐⭐(arXiv preprint 2022) *Compositional Visual Generation with Composable Diffusion Models*, Nan Liu et al. [[Paper](https://arxiv.org/abs/2206.01714)] [[Code](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch)] [[Project](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/)]
    * 🍬 This method is ***an alternative structured approach for compositional generation using diffusion models***. An image is generated by composing a set of diffusion models, with each of them modeling a certain component of the image. 
* ⭐⭐(arXiv preprint 2022) [Imagen] *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding*, Chitwan Saharia et al. [[Paper](https://arxiv.org/abs/2205.11487)] [[Blog](https://gweb-research-imagen.appspot.com/)]
    * 🍬 Imagen: ***A text-to-image diffusion model with an unprecedented degree of photorealism and a deep level of language understanding***, which builds on the power of ***large transformer language models in understanding text*** and hinges on the strength of ***diffusion models in high-fidelity image generation***. 
* ⭐(OpenAI) [DALL-E 2] *Hierarchical Text-Conditional Image Generation with CLIP Latents*, Aditya Ramesh et al. [[Paper](https://cdn.openai.com/papers/dall-e-2.pdf)] [[Blog](https://openai.com/dall-e-2/)] [[Risks and Limitations](https://github.com/openai/dalle-2-preview/blob/main/system-card.md)] [[Unofficial Code](https://github.com/lucidrains/DALLE2-pytorch)] 
    * 🍬 ***DALL-E 2***: A two-stage model, a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditioned on the image embedding.
(2022.08.06 Update)
* ⭐(arXiv preprint 2022) *CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers*, Wenyi Hong et al. [[Paper](https://arxiv.org/abs/2205.15868)] [[Code](https://github.com/THUDM/CogVideo)]
    * 🍬 CogVideo: ***The first open-source large-scale pretrained text-to-video model***, which is trained by ***inheriting a pretrained text-to-image model*** (CogView2) and outperforms all publicly available models at a large margin in machine and human evaluations.
 -->

## <span id="head6"> *6. Paper With Code* </span>

* <span id="head-Survey"> **Survey**  </span> **[       «🎯Back To Top»       ](#)**
    * **Text-to-Image Synthesis: A Comparative Study** [[v1](https://link.springer.com/chapter/10.1007/978-981-16-2275-5_14)(Digital Transformation Technology)] (2021.08) 
    * **A survey on generative adversarial network-based text-to-image synthesis** [[v1](https://www.sciencedirect.com/science/article/pii/S0925231221006111)(Neurocomputing)] (2021.04) 
    * **Adversarial Text-to-Image Synthesis: A Review** [[v1](https://arxiv.org/abs/2101.09983v1)(arXiv)] (2021.01) [[v2](https://www.sciencedirect.com/science/article/pii/S0893608021002823)(Neural Networks)] (2021.08)
    * **A Survey and Taxonomy of Adversarial Neural Networks for Text-to-Image Synthesis** [[v1](https://arxiv.org/pdf/1910.09399.pdf)(arXiv)] (2019.10) 

* <span id="head-T2F"> **Text to Face👨🏻🧒👧🏼🧓🏽**  </span> **[       «🎯Back To Top»       ](#)**
    * (arXiv preprint 2022) **Bridging CLIP and StyleGAN through Latent Alignment for Image Editing**, Wanfeng Zheng et al. [[Paper](https://arxiv.org/abs/2210.04506)] 
    * (ACMMM 2022) **Learning Dynamic Prior Knowledge for Text-to-Face Pixel Synthesis**, Jun Peng et al. [[Paper](https://dl.acm.org/doi/10.1145/3503161.3547818)]
    * (ACMMM 2022) **Towards Open-Ended Text-to-Face Generation, Combination and Manipulation**, Jun Peng et al. [[Paper](https://dl.acm.org/doi/abs/10.1145/3503161.3547758)]
    * (BMVC 2022) **clip2latent: Text driven sampling of a pre-trained StyleGAN using denoising diffusion and CLIP**, Justin N. M. Pinkney et al. [[Paper](https://arxiv.org/abs/2210.02347v1)] [[Code](https://github.com/justinpinkney/clip2latent)]
    * (arXiv preprint 2022) **ManiCLIP: Multi-Attribute Face Manipulation from Text**, Hao Wang et al. [[Paper](https://arxiv.org/abs/2210.00445)]
    * (arXiv preprint 2022) **Generated Faces in the Wild: Quantitative Comparison of Stable Diffusion, Midjourney and DALL-E 2**, Ali Borji, [[Paper](https://arxiv.org/abs/2210.00586)] [[Code](https://github.com/aliborji/GFW)] [[Data](https://drive.google.com/file/d/1EhbUK64J3d0_chmD2mpBuWB-Ic7LeFlP/view)]
    * (arXiv preprint 2022) **Text-Free Learning of a Natural Language Interface for Pretrained Face Generators**, Xiaodan Du et al. [[Paper](https://arxiv.org/abs/2209.03953)] [[Code](https://github.com/duxiaodan/Fast_text2StyleGAN)]
    * (Knowledge-Based Systems-2022) **CMAFGAN: A Cross-Modal Attention Fusion based Generative Adversarial Network for attribute word-to-face synthesis**, Xiaodong Luo et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S0950705122008863)]
    * (Neural Networks-2022) **DualG-GAN, a Dual-channel Generator based Generative Adversarial Network for text-to-face synthesis**, Xiaodong Luo et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S0893608022003161)]
    * (arXiv preprint 2022) **Text-to-Face Generation with StyleGAN2**, D. M. A. Ayanthi et al. [[Paper](https://arxiv.org/abs/2205.12512)]
    * (CVPR 2022) **StyleT2I: Toward Compositional and High-Fidelity Text-to-Image Synthesis**, Zhiheng Li et al. [[Paper](https://arxiv.org/abs/2203.15799)] [[Code](https://github.com/zhihengli-UR/StyleT2I)]
    * (arXiv preprint 2022) **StyleT2F: Generating Human Faces from Textual Description Using StyleGAN2**, Mohamed Shawky Sabae et al. [[Paper](https://arxiv.org/abs/2204.07924)] [[Code](https://github.com/DarkGeekMS/Retratista)]
    * (CVPR 2022) **AnyFace: Free-style Text-to-Face Synthesis and Manipulation**, Jianxin Sun et al. [[Paper](https://arxiv.org/abs/2203.15334)] 
    * (IEEE Transactions on Network Science and Engineering-2022) **TextFace: Text-to-Style Mapping based Face Generation and Manipulation**, Xianxu Hou et al. [[Paper](https://ieeexplore.ieee.org/abstract/document/9737433)]
    * (FG 2021) **Generative Adversarial Network for Text-to-Face Synthesis and Manipulation with Pretrained BERT Model**, Yutong Zhou et al. [[Paper](https://ieeexplore.ieee.org/document/9666791)] 
    * (ACMMM 2021) **Multi-caption Text-to-Face Synthesis: Dataset and Algorithm**, Jianxin Sun et al. [[Paper](https://dl.acm.org/doi/10.1145/3474085.3475391)] [[Code](https://github.com/cripac-sjx/SEA-T2F)]
    * (ACMMM 2021) **Generative Adversarial Network for Text-to-Face Synthesis and Manipulation**, Yutong Zhou. [[Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3481026)]
    * (WACV 2021) **Faces a la Carte: Text-to-Face Generation via Attribute Disentanglement**, Tianren Wang et al. [[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Wang_Faces_a_la_Carte_Text-to-Face_Generation_via_Attribute_Disentanglement_WACV_2021_paper.pdf)] 
    * (arXiv preprint 2019) **FTGAN: A Fully-trained Generative Adversarial Networks for Text to Face Generation**, Xiang Chen et al. [[Paper](https://arxiv.org/abs/1904.05729)]

* <span id="head-CI"> **Compounding Issues🤔**  </span> **[       «🎯Back To Top»       ](#)**
    * (arXiv preprint 2022) [💬 Interpretable Intervention] **Not Just Pretty Pictures: Text-to-Image Generators Enable Interpretable Interventions for Robust Representations**, Jianhao Yuan et al.  [[Paper](https://arxiv.org/abs/2212.11237)]
    * (arXiv preprint 2022) [💬 Ethical Image Manipulation] **Judge, Localize, and Edit: Ensuring Visual Commonsense Morality for Text-to-Image Generation**, Seongbeom Park et al.  [[Paper](https://arxiv.org/abs/2212.03507)]
    * (arXiv preprint 2022) [💬 Creativity Transfer] **Inversion-Based Creativity Transfer with Diffusion Models**, Yuxin Zhang et al.  [[Paper](https://arxiv.org/abs/2211.13203)]
    * (arXiv preprint 2022) [💬 Ambiguity] **Is the Elephant Flying? Resolving Ambiguities in Text-to-Image Generative Models**, Ninareh Mehrabi et al.  [[Paper](https://arxiv.org/abs/2211.12503)]
    * (arXiv preprint 2022) [💬 Racial Politics] **A Sign That Spells: DALL-E 2, Invisual Images and The Racial Politics of Feature Space**, Fabian Offert et al.  [[Paper](https://arxiv.org/abs/2211.06323)]
    * (arXiv preprint 2022) [💬 Demographic Stereotypes] **Easily Accessible Text-to-Image Generation Amplifies Demographic Stereotypes at Large Scale**, Federico Bianchi et al.  [[Paper](https://arxiv.org/abs/2211.03759)]
    * (arXiv preprint 2022) [💬 Privacy Analysis] **Membership Inference Attacks Against Text-to-image Generation Models**, Yixin Wu et al.  [[Paper](https://arxiv.org/abs/2210.00968)]
    * (arXiv preprint 2022) [💬 Authenticity Evaluation for Fake Images] **DE-FAKE: Detection and Attribution of Fake Images Generated by Text-to-Image Diffusion Models**, Zeyang Sha et al.  [[Paper](https://arxiv.org/abs/2210.06998v1)] 
    * (arXiv preprint 2022) [💬 Cultural Bias] **The Biased Artist: Exploiting Cultural Biases via Homoglyphs in Text-Guided Image Generation Models**, Lukas Struppek et al.  [[Paper](https://arxiv.org/abs/2209.08891)] 

* <span id="head-2023"> **2023**  </span> **[       «🎯Back To Top»       ](#)**
     * (IEEE Transactions on Multimedia) **ALR-GAN: Adaptive Layout Refinement for Text-to-Image Synthesis**, Hongchen Tan et al. [[Paper](https://ieeexplore.ieee.org/abstract/document/10023990)] 
     * (arXiv preprint 2023) **GLIGEN: Open-Set Grounded Text-to-Image Generation**, Yuheng Li et al. [[Paper](https://arxiv.org/abs/2301.07093)] [[Project](https://gligen.github.io/)] [[Hugging Face Demo](https://huggingface.co/spaces/gligen/demo)] 
     * (arXiv preprint 2023) **Attribute-Centric Compositional Text-to-Image Generation**, Yuren Cong et al. [[Paper](https://arxiv.org/abs/2301.01413)] [[Project](https://github.com/yrcong/ACTIG)] 
     * ⭐⭐(arXiv preprint 2023) **Muse: Text-To-Image Generation via Masked Generative Transformers**, Huiwen Chang et al. [[Paper](https://arxiv.org/abs/2301.00704v1)] [[Project](https://muse-model.github.io/)] 

* <span id="head-2022"> **2022**  </span> **[       «🎯Back To Top»       ](#)**
     * ⭐(arXiv preprint 2022) **Multi-Concept Customization of Text-to-Image Diffusion**, Nupur Kumari et al.  [[Paper](https://arxiv.org/abs/2212.04488)] [[Project](https://www.cs.cmu.edu/~custom-diffusion/)] [[Code](https://github.com/adobe-research/custom-diffusion)] [[Hugging Face](https://huggingface.co/spaces/nupurkmr9/custom-diffusion)]
     * (arXiv preprint 2022) [💬Optimizing Prompts] **Optimizing Prompts for Text-to-Image Generation**, Yaru Hao et al. [[Paper](https://arxiv.org/abs/2212.09611)] [[Code](https://github.com/microsoft/LMOps)] 
     * (arXiv preprint 2022) [💬Spatial Relationship] **Benchmarking Spatial Relationships in Text-to-Image Generation**, Tejas Gokhale et al. [[Paper](https://arxiv.org/abs/2212.10015)]
     * ⭐(arXiv preprint 2022) **Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis**, Weixi Feng et al.  [[Paper](https://arxiv.org/abs/2212.05032)] [[Code](https://github.com/shunk031/training-free-structured-diffusion-guidance)]
     * (arXiv preprint 2022) [💬Multilingual Communication System] **Multilingual Communication System with Deaf Individuals Utilizing Natural and Visual Languages**, Tuan-Luc Huynh et al.  [[Paper](https://arxiv.org/abs/2212.00305v1)]
     * (arXiv preprint 2022) [💬Controllable] **SpaText: Spatio-Textual Representation for Controllable Image Generation**, Omri Avrahami et al.  [[Paper](https://arxiv.org/abs/2211.14305)] [[Project](https://omriavrahami.com/spatext/)]
     * (arXiv preprint 2022) [💬Human Evaluation] **Human Evaluation of Text-to-Image Models on a Multi-Task Benchmark**, Vitali Petsiuk et al.  [[Paper](https://arxiv.org/abs/2211.12112)]
    * (arXiv preprint 2022) **HumanDiffusion: a Coarse-to-Fine Alignment Diffusion Framework for Controllable Text-Driven Person Image Generation**, Kaiduo Zhang et al.  [[Paper](https://arxiv.org/abs/2211.06235)]
    * ⭐⭐(arXiv preprint 2022) **eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers**, Yogesh Balaji et al.  [[Paper](https://arxiv.org/abs/2211.01324)] [[Project](https://deepimagination.cc/eDiffi/)] [[Video](https://www.youtube.com/watch?v=k6cOx9YjHJc)]
    * (arXiv preprint 2022) [💬Text-Image Consistency] **Towards Better Text-Image Consistency in Text-to-Image Generation**, Zhaorui Tan et al.  [[Paper](https://arxiv.org/abs/2210.15235)] 
    * (arXiv preprint 2022) **ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts**, Zhida Feng et al.  [[Paper](https://arxiv.org/abs/2210.15257)]
    * ⭐(ECCV 2022) [💬Evaluation Metrics] **TISE: Bag of Metrics for Text-to-Image Synthesis Evaluation**, Tan M. Dinh et al.  [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960585.pdf)] [[Code](https://github.com/VinAIResearch/tise-toolbox)] [[Project](https://di-mi-ta.github.io/tise/)]
    * (ECCV 2022) [💬Trace+Text→Image] **Trace Controlled Text to Image Generation**, Kun Yan et al.  [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960058.pdf)]
    * (arXiv preprint 2022) [💬Markup→Image] **Markup-to-Image Diffusion Models with Scheduled Sampling**, Yuntian Deng et al.  [[Paper](https://arxiv.org/abs/2210.05147)] [[Code](https://github.com/da03/markup2im)]
    * (arXiv preprint 2022) **Swinv2-Imagen: Hierarchical Vision Transformer Diffusion Models for Text-to-Image Generation**, Ruijun Li et al.  [[Paper](https://arxiv.org/abs/2210.09549)]
    * ⭐⭐(arXiv preprint 2022) **DALL-E-Bot: Introducing Web-Scale Diffusion Models to Robotics**, Ivan Kapelyukh et al.  [[Paper](https://arxiv.org/abs/2210.02438v1)] [[Project](https://www.robot-learning.uk/dall-e-bot)]
    * (arXiv preprint 2022) **Progressive Denoising Model for Fine-Grained Text-to-Image Generation**, Zhengcong Fei et al.  [[Paper](https://arxiv.org/abs/2210.02291v1)] 
    * (arXiv preprint 2022) **Creative Painting with Latent Diffusion Models**, Xianchao Wu  [[Paper](https://arxiv.org/abs/2209.14697)] 
    * (arXiv preprint 2022) **Re-Imagen: Retrieval-Augmented Text-to-Image Generator**, Wenhu Chen et al.  [[Paper](https://arxiv.org/abs/2209.14491)] 
    * (ACMMM 2022) **AtHom: Two Divergent Attentions Stimulated By Homomorphic Training in Text-to-Image Synthesis**, Zhenbo Shi et al. [[Paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548159)]
    * (ACMMM 2022) **Adma-GAN: Attribute-Driven Memory Augmented GANs for Text-to-Image Generation**, Xintian Wu et al.  [[Paper](https://arxiv.org/abs/2209.14046)] [[Code](https://github.com/Hsintien-Ng/Adma-GAN)]
    * (arXiv preprint 2022) [💬Aesthetic Image Generation] **Best Prompts for Text-to-Image Models and How to Find Them**, Nikita Pavlichenko et al.  [[Paper](https://arxiv.org/abs/2209.11711)] 
    * (ACMMM 2022) **AI Illustrator: Translating Raw Descriptions into Images by Prompt-based Cross-Modal Generation**, Yiyang Ma et al.  [[Paper](https://arxiv.org/abs/2209.03160v1)] 
    * (ACMMM 2022) **DSE-GAN: Dynamic Semantic Evolution Generative Adversarial Network for Text-to-Image Generation**, Mengqi Huang et al.  [[Paper](https://dl.acm.org/doi/10.1145/3503161.3547881)] 
    * (arXiv preprint 2022) [💬Radiology] **What Does DALL-E 2 Know About Radiology?**, Lisa C. Adams et al.  [[Paper](https://arxiv.org/abs/2209.13696)] 
    * (arXiv preprint 2022) **Prompt-to-Prompt Image Editing with Cross Attention Control**, Amir Hertz et al.  [[Paper](https://arxiv.org/abs/2208.01626)] [[Code](https://github.com/google/prompt-to-prompt)] [[Unofficial Code](https://github.com/bloc97/CrossAttentionControl)] [[Project](https://prompt-to-prompt.github.io/)]
    * (arXiv preprint 2022) **Text to Image Generation: Leaving no Language Behind**, Pedro Reviriego et al.  [[Paper](https://arxiv.org/abs/2208.09333)] 
    * (arXiv preprint 2022) **Vision-Language Matching for Text-to-Image Synthesis via Generative Adversarial Networks**, Qingrong Cheng et al.  [[Paper](https://arxiv.org/abs/2208.09596)] 
    * (arXiv preprint 2022) [💬Visual Understanding on Generated Images] **How good are deep models in understanding the generated images?**, Ali Borji  [[Paper](https://arxiv.org/abs/2208.10760)] 
    * (arXiv preprint 2022) **Text-to-Image Generation via Implicit Visual Guidance and Hypernetwork**, Xin Yuan et al.  [[Paper](https://arxiv.org/abs/2208.08493)] 
    * (arXiv preprint 2022) [💬Hybrid word→Image] **Adversarial Attacks on Image Generation With Made-Up Words**, Raphaël Millière  [[Paper](https://arxiv.org/abs/2208.04135)] 
    * (arXiv preprint 2022) **Memory-Driven Text-to-Image Generation**, Bowen Li et al.  [[Paper](https://arxiv.org/abs/2208.07022)] 
    * (arXiv preprint 2022) [💬Text-to-Person]**T-Person-GAN: Text-to-Person Image Generation with Identity-Consistency and Manifold Mix-Up**, Lin Wu et al.  [[Paper](https://arxiv.org/abs/2208.12752)] [[Code](https://github.com/linwu-github/Person-Image-Generation)]
    * (arXiv preprint 2022) **LogicRank: Logic Induced Reranking for Generative Text-to-Image Systems**, Björn Deiseroth et al.  [[Paper](https://arxiv.org/abs/2208.13518)] 
    * (arXiv preprint 2022) [💬Text→Layout→Image] **Layout-Bridging Text-to-Image Synthesis**, Jiadong Liang et al.  [[Paper](https://arxiv.org/abs/2208.06162)] 
    * (arXiv preprint 2022) **DALLE-URBAN: Capturing the urban design expertise of large text to image transformers**, Sachith Seneviratne et al.  [[Paper](https://arxiv.org/abs/2208.04139)] [[Generated Images](https://github.com/sachith500/DALLEURBAN)] 
    * (arXiv preprint 2022) **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion**, Rinon Gal et al.  [[Paper](https://arxiv.org/abs/2208.01618)] [[Code](https://github.com/rinongal/textual_inversion)] [[Project](https://textual-inversion.github.io/)]
    * (arXiv preprint 2022) [💬Relational Understanding Analysis] **Testing Relational Understanding in Text-Guided Image Generation**, Colin Conwell  [[Paper](https://arxiv.org/abs/2208.00005)] 
    * (arXiv preprint 2022) [💬Lighting Consistency Analysis] **Lighting (In)consistency of Paint by Text**, Hany Farid  [[Paper](https://arxiv.org/abs/2207.13744)] 
    * (ECCV 2022) **NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion**, Chenfei Wu et al. [[Paper](https://arxiv.org/abs/2111.12417)] [[Code](https://github.com/microsoft/NUWA)]
        * **Multimodal Pretrained Model for Multi-tasks🎄**: Text-To-Image (T2I), Sketch-to-Image (S2I), Image Completion (I2I), Text-Guided Image Manipulation (TI2I), Text-to-Video (T2V), Video Prediction (V2V), Sketch-to-Video (S2V), Text-Guided Video Manipulation (TV2V)
          ![Figure from paper](pic/NUWA.gif)
          > *(From: https://github.com/microsoft/NUWA [2021/11/30])*
    * (ECCV 2022) **Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors**, Oran Gafni et al. [[Paper](https://arxiv.org/abs/2203.13131)] [[Code](https://github.com/CasualGANPapers/Make-A-Scene)] [[The Little Red Boat Story](https://www.youtube.com/watch?v=N4BagnXzPXY)]
    * (arXiv preprint 2022) **Exploring Generative Adversarial Networks for Text-to-Image Generation with Evolution Strategies**, Victor Costa et al.  [[Paper](https://arxiv.org/abs/2207.02907)] 
    * (CVPR 2022) **Text to Image Generation with Semantic-Spatial Aware GAN**, Wentong Liao et al. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liao_Text_to_Image_Generation_With_Semantic-Spatial_Aware_GAN_CVPR_2022_paper.pdf)] [[Code](https://github.com/wtliao/text2image)]
    * (ICMR 2022) **Disentangled Representations and Hierarchical Refinement of Multi-Granularity Features for Text-to-Image Synthesis**, Pei Dong et al.  [[Paper](https://dl.acm.org/doi/abs/10.1145/3512527.3531389)] 
    * (arXiv preprint 2022) **Scaling Autoregressive Models for Content-Rich Text-to-Image Generation**, Jiahui Yu et al.  [[Paper](https://arxiv.org/abs/2206.10789)] [[Code](https://github.com/google-research/parti)] [[Project](https://parti.research.google/)]
    * (Information Sciences-2022) **Text-to-Image Synthesis: Starting Composite from the Foreground Content**, Zhiqiang Zhang et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S0020025522006399)]
    * (Applied Intelligence-2022) **Generative adversarial network based on semantic consistency for text-to-image generation**, Yue Ma et al. [[Paper](https://link.springer.com/article/10.1007/s10489-022-03660-8)]
    * (ICML 2022) **GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models**, Alex Nichol et al. [[Paper](https://arxiv.org/abs/2112.10741)] [[Code](https://github.com/openai/glide-text2im)]
    * ⭐⭐(arXiv preprint 2022) **Compositional Visual Generation with Composable Diffusion Models**, Nan Liu et al. [[Paper](https://arxiv.org/abs/2206.01714)] [[Code](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch)] [[Project](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/)]
    * (SIGGRAPH 2022) **Text2Human: Text-Driven Controllable Human Image Generation**, Yuming Jiang et al. [[Paper](https://arxiv.org/pdf/2205.15996.pdf)] [[Code](https://github.com/yumingj/Text2Human)]
    * ⭐⭐(arXiv preprint 2022) [Imagen] **Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding**, Chitwan Saharia et al. [[Paper](https://arxiv.org/abs/2205.11487)] [[Blog](https://gweb-research-imagen.appspot.com/)]
    * (ICME 2022) **GR-GAN: Gradual Refinement Text-to-image Generation**, Bo Yang et al. [[Paper](https://arxiv.org/abs/2205.11273)] [[Code](https://github.com/BoO-18/GR-GAN)]
    * (CHI 2022) **Design Guidelines for Prompt Engineering Text-to-Image Generative Models**, Vivian Liu et al. [[Paper](https://dl.acm.org/doi/10.1145/3491102.3501825)]
    * (Neural Processing Letters-2022) **PBGN: Phased Bidirectional Generation Network in Text-to-Image Synthesis**, Jianwei Zhu et al. [[Paper](https://link.springer.com/article/10.1007/s11063-022-10866-x)]
    * (Signal Processing: Image Communication-2022) **ARRPNGAN: Text-to-image GAN with attention regularization and region proposal networks**, Fengnan Quan et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S0923596522000601)] [[Code](https://github.com/quanFN/ARRPNGAN)]
    * (arXiv preprint 2022) **CogView2: Faster and Better Text-to-Image Generation via Hierarchical Transformers**, Ming Ding et al. [[Paper](https://arxiv.org/abs/2204.14217)] [[Code](https://github.com/THUDM/CogView2)]
    * ⭐(OpenAI) [DALL-E 2] **Hierarchical Text-Conditional Image Generation with CLIP Latents**, Aditya Ramesh et al. [[Paper](https://cdn.openai.com/papers/dall-e-2.pdf)] [[Blog](https://openai.com/dall-e-2/)] [[Risks and Limitations](https://github.com/openai/dalle-2-preview/blob/main/system-card.md)] [[Unofficial Code](https://github.com/lucidrains/DALLE2-pytorch)] 
    * (arXiv preprint 2022) **Recurrent Affine Transformation for Text-to-image Synthesis**, Senmao Ye et al. [[Paper](https://arxiv.org/abs/2204.10482)] [[Code](https://github.com/senmaoy/Recurrent-Affine-Transformation-for-Text-to-image-Synthesis)]
    * (AAAI 2022) **Interactive Image Generation with Natural-Language Feedback**, Yufan Zhou et al. [[Paper](https://www.aaai.org/AAAI22Papers/AAAI-7081.ZhouY.pdf)]
    * (IEEE Transactions on Neural Networks and Learning Systems-2022) **DR-GAN: Distribution Regularization for Text-to-Image Generation**, Hongchen Tan et al. [[Paper](https://arxiv.org/abs/2204.07945)] [[Code](https://github.com/Tan-H-C/DR-GAN-Distribution-Regularization-for-Text-to-Image-Generation)]
    * (Pattern Recognition Letters-2022) **Text-to-image synthesis with self-supervised learning**, Yong Xuan Tan et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S0167865522001064)] 
    * (CVPR 2022) **Vector Quantized Diffusion Model for Text-to-Image Synthesis**, Shuyang Gu et al. [[Paper](https://arxiv.org/abs/2111.14822)] [[Code](https://github.com/microsoft/vq-diffusion)]
    * (CVPR 2022) **Autoregressive Image Generation using Residual Quantization**, Doyup Lee et al. [[Paper](https://arxiv.org/abs/2203.01941)] [[Code](https://github.com/kakaobrain/rq-vae-transformer)] 
    * (CVPR 2022) **Text-to-Image Synthesis based on Object-Guided Joint-Decoding Transformer**, Fuxiang Wu et al. [[Paper](https://fengxianghe.github.io/paper/wu2022text.pdf)]
    * (CVPR 2022) **LAFITE: Towards Language-Free Training for Text-to-Image Generation**, Yufan Zhou et al. [[Paper](https://arxiv.org/abs/2111.13792)] [[Code](https://github.com/drboog/Lafite)] 
    * (CVPR 2022) **DF-GAN:  A Simple and Effective Baseline for Text-to-Image Synthesis**, Ming Tao et al. [[Paper](https://arxiv.org/abs/2008.05865)] [[Code](https://github.com/tobran/DF-GAN)] 
    * (arXiv preprint 2022) **DT2I: Dense Text-to-Image Generation from Region Descriptions**, Stanislav Frolov et al. [[Paper](https://arxiv.org/abs/2204.02035)] 
    * (arXiv preprint 2022) **CLIP-GEN: Language-Free Training of a Text-to-Image Generator with CLIP**, Zihao Wang et al. [[Paper](https://arxiv.org/abs/2203.00386)] [[Code](https://github.com/HFAiLab/clip-gen)]
    * (arXiv preprint 2022) **OptGAN: Optimizing and Interpreting the Latent Space of the Conditional Text-to-Image GANs**, Zhenxing Zhang et al. [[Paper](https://arxiv.org/abs/2202.12929)]
    * (arXiv preprint 2022) **DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers**, Jaemin Cho et al. [[Paper](https://arxiv.org/abs/2202.04053)] [[Code](https://github.com/j-min/DallEval)] 
    * (IEEE Transactions on Network Science and Engineering-2022) **Neural Architecture Search with a Lightweight Transformer for Text-to-Image Synthesis**, Wei Li et al. [[Paper](https://ieeexplore.ieee.org/abstract/document/9699403)] 
    * (Neurocomputing-2022) **DiverGAN: An Efficient and Effective Single-Stage Framework for Diverse Text-to-Image Generation**, Zhenxing Zhang et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231221018397)]
    * (Knowledge-Based Systems-2022) **CJE-TIG: Zero-shot cross-lingual text-to-image generation by Corpora-based Joint Encoding**, Han Zhang et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S0950705121011138)] 
    * (WACV 2022) **StyleMC: Multi-Channel Based Fast Text-Guided Image Generationand Manipulation**, Umut Kocasarı et al. [[Paper](https://arxiv.org/abs/2112.08493)] [[Project](https://catlab-team.github.io/stylemc/)]

* <span id="head-2021"> **2021**  </span> **[       «🎯Back To Top»       ](#)**
    * (arXiv preprint 2021) **Multimodal Conditional Image Synthesis with Product-of-Experts GANs**, Xun Huang et al. [[Paper](https://arxiv.org/abs/2112.05130)]  [[Project](https://deepimagination.cc/PoE-GAN/)]
         * Text-to-Image, Segmentation-to-Image, *Text+Segmentation/Sketch/Image→Image*, *Sketch+Segmentation/Image→Image*, *Segmentation+Image→Image*
    * (IEEE TCSVT) **RiFeGAN2: Rich Feature Generation for Text-to-Image Synthesis from Constrained Prior Knowledge**, Jun Cheng et al. [[Paper](https://ieeexplore.ieee.org/abstract/document/9656731)] 
    * (ICONIP 2021) **TRGAN: Text to Image Generation Through Optimizing Initial Image**, Liang Zhao et al. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-92307-5_76)] 
    * (NeurIPS 2021) **Benchmark for Compositional Text-to-Image Synthesis**, Dong Huk Park et al. [[Paper](https://openreview.net/forum?id=bKBhQhPeKaF)] [[Code](https://github.com/Seth-Park/comp-t2i-dataset)]
    * (arXiv preprint 2021) **FuseDream: Training-Free Text-to-Image Generation with Improved CLIP+GAN Space Optimization**, Xingchao Liu et al. [[Paper](https://arxiv.org/abs/2112.01573)] [[Code](https://github.com/gnobitab/FuseDream)]
    * (ICONIP 2021) **Self-Supervised Image-to-Text and Text-to-Image Synthesis**, Anindya Sundar Das et al. [[Paper](https://arxiv.org/abs/2112.04928)]
    * (arXiv preprint 2021) **DiverGAN: An Efficient and Effective Single-Stage Framework for Diverse Text-to-Image Generation**, Zhenxing Zhang et al. [[Paper](https://arxiv.org/pdf/2111.09267.pdf)] 
    * (Image and Vision Computing) **Transformer models for enhancing AttnGAN based text to image generation**, S. Naveen et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S026288562100189X)]
    * (ACMMM 2021) **R-GAN: Exploring Human-like Way for Reasonable Text-to-Image Synthesis via Generative Adversarial Networks**, Yanyuan Qiao et al. [[Paper](https://dl.acm.org/doi/10.1145/3474085.3475363)]
    * (ACMMM 2021) **Cycle-Consistent Inverse GAN for Text-to-Image Synthesis**, Hao Wang et al. [[Paper](https://dl.acm.org/doi/10.1145/3474085.3475226)]
    * (ACMMM 2021) **Unifying Multimodal Transformer for Bi-directional Image and Text Generation**, Yupan Huang et al. [[Paper](https://dl.acm.org/doi/10.1145/3474085.3481540)] [[Code](https://github.com/researchmm/generate-it)]
    * (ACMMM 2021) **A Picture is Worth a Thousand Words: A Unified System for Diverse Captions and Rich Images Generation**, Yupan Huang et al. [[Paper](https://dl.acm.org/doi/10.1145/3474085.3478561)] [[Code](https://github.com/researchmm/generate-it)]
    * (ICCV 2021) **Talk-to-Edit: Fine-Grained Facial Editing via Dialog**, Yuming Jiang et al. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Talk-To-Edit_Fine-Grained_Facial_Editing_via_Dialog_ICCV_2021_paper.pdf)] [[Project](https://www.mmlab-ntu.com/project/talkedit/)] [[Code](https://github.com/yumingj/Talk-to-Edit)]
    * (ICCV 2021) **DAE-GAN: Dynamic Aspect-Aware GAN for Text-to-Image Synthesis**, Shulan Ruan et al. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ruan_DAE-GAN_Dynamic_Aspect-Aware_GAN_for_Text-to-Image_Synthesis_ICCV_2021_paper.pdf)] [[Supp](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Ruan_DAE-GAN_Dynamic_Aspect-Aware_ICCV_2021_supplemental.pdf)] [[Code](https://github.com/hiarsal/DAE-GAN)]
    * (ICIP 2021) **Text To Image Synthesis With Erudite Generative Adversarial Networks**, Zhiqiang Zhang et al. [[Paper](https://ieeexplore.ieee.org/document/9506487)] 
    * (PRCV 2021) **MAGAN: Multi-attention Generative Adversarial Networks for Text-to-Image Generation**, Xibin Jia et al. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-88013-2_26)]
    * (AAAI 2021) **TIME: Text and Image Mutual-Translation Adversarial Networks**, Bingchen Liu et al. [[Paper](https://www.aaai.org/AAAI21Papers/AAAI-1426.LiuB.pdf)] [[arXiv Paper](https://arxiv.org/pdf/2005.13192.pdf)] 
    * (IJCNN 2021) **Text to Image Synthesis based on Multi-Perspective Fusion**, Zhiqiang Zhang et al. [[Paper](https://ieeexplore.ieee.org/document/9533925)] 
    * (arXiv preprint 2021) **CRD-CGAN: Category-Consistent and Relativistic Constraints for Diverse Text-to-Image Generation**, Tao Hu et al. [[Paper](https://arxiv.org/abs/2107.13516)]
    * (arXiv preprint 2021) **Improving Text-to-Image Synthesis Using Contrastive Learning**, Hui Ye et al. [[Paper](https://arxiv.org/pdf/2107.02423v1.pdf)] [[Code](https://github.com/huiyegit/T2I_CL)]
    * (arXiv preprint 2021) **CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders**, Kevin Frans et al. [[Paper](https://arxiv.org/pdf/2106.14843.pdf)] [[Code](https://colab.research.google.com/github/kvfrans/clipdraw/blob/main/clipdraw.ipynb)]
    * (ICASSP 2021) **Drawgan: Text to Image Synthesis with Drawing Generative Adversarial Networks**, Zhiqiang Zhang et al. [[Paper](https://ieeexplore.ieee.org/document/9414166)] 
    * (IJCNN 2021) **DTGAN: Dual Attention Generative Adversarial Networks for Text-to-Image Generation**, Zhenxing Zhang et al. [[Paper](https://ieeexplore.ieee.org/abstract/document/9533527)] 
    * (CVPR 2021) **TediGAN: Text-Guided Diverse Image Generation and Manipulation**, Weihao Xia et al. [[Paper](https://arxiv.org/pdf/2012.03308.pdf)] [[Extended Version](https://arxiv.org/pdf/2104.08910.pdf)][[Code](https://github.com/IIGROUP/TediGAN)] [[Dataset](https://github.com/IIGROUP/Multi-Modal-CelebA-HQ-Dataset)] [[Colab](https://colab.research.google.com/github/weihaox/TediGAN/blob/main/playground.ipynb)] [[Video](https://www.youtube.com/watch?v=L8Na2f5viAM)] 
    * (CVPR 2021) **Cross-Modal Contrastive Learning for Text-to-Image Generation**, Han Zhang et al. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Cross-Modal_Contrastive_Learning_for_Text-to-Image_Generation_CVPR_2021_paper.pdf)] [[Code](https://github.com/google-research/xmcgan_image_generation)]
    * (NeurIPS 2021) **CogView: Mastering Text-to-Image Generation via Transformers**, Ming Ding et al. [[Paper](https://arxiv.org/pdf/2105.13290.pdf)] [[Code](https://github.com/THUDM/CogView)] [[Demo Website(Chinese)](https://lab.aminer.cn/cogview/index.html)] 
    * (IEEE Transactions on Multimedia 2021) **Modality Disentangled Discriminator for Text-to-Image Synthesis**, Fangxiang Feng et al. [[Paper](https://ieeexplore.ieee.org/document/9417738)] [[Code](https://github.com/FangxiangFeng/DM-GAN-MDD)]
    * ⭐(arXiv preprint 2021) **Zero-Shot Text-to-Image Generation**, Aditya Ramesh et al. [[Paper](https://arxiv.org/pdf/2102.12092.pdf)] [[Code](https://github.com/openai/DALL-E)] [[Blog](https://openai.com/blog/dall-e/)] [[Model Card](https://github.com/openai/DALL-E/blob/master/model_card.md)] [[Colab](https://colab.research.google.com/github/openai/DALL-E/blob/master/notebooks/usage.ipynb)] [[Code(Pytorch)](https://github.com/lucidrains/DALLE-pytorch)]
    <!--https://colab.research.google.com/drive/1KA2w8bA9Q1HDiZf5Ow_VNOrTaWW4lXXG?usp=sharing -->
    * (Pattern Recognition 2021) **Unsupervised text-to-image synthesis**, Yanlong Dong et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320320303769)] 
    * (WACV 2021) **Text-to-Image Generation Grounded by Fine-Grained User Attention**, Jing Yu Koh et al. [[Paper](https://arxiv.org/pdf/2011.03775.pdf)] [[Code](https://github.com/google-research/trecs_image_generation)]
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


## <span id="head7"> *7. Other Related Works* </span>
   * <span id="head-MM"> **⭐Multimodality⭐** </span> **[       «🎯Back To Top»       ](#)**
       * (arXiv preprint 2023) **Muse: Text-To-Image Generation via Masked Generative Transformers**, Huiwen Chang et al. [[Paper](https://arxiv.org/abs/2301.00704v1)] [[Project](https://muse-model.github.io/)] 
         * 📚Text-to-Image, Zero-shot+Mask-free editing, Zero-shot Inpainting/Outpainting
       * (arXiv preprint 2022) **Versatile Diffusion: Text, Images and Variations All in One Diffusion Model**, Xingqian Xu et al. [[Paper](https://arxiv.org/abs/2211.08332)] [[Code](https://github.com/SHI-Labs/Versatile-Diffusion)] [[Hugging Face](https://huggingface.co/spaces/shi-labs/Versatile-Diffusion)]
         * 📚Text-to-Image, Image-Variation, Image-to-Text, Disentanglement, Text+Image-Guided Generation, Editable I2T2I
       * (arXiv preprint 2022) **Frido: Feature Pyramid Diffusion for Complex Scene Image Synthesis**, Wan-Cyuan Fan et al. [[Paper](https://arxiv.org/abs/2208.13753)] [[Code](https://github.com/davidhalladay/Frido)]
         * 📚Text-to-Image, Scene Gragh to Image, Layout-to-Image, Uncondition Image Generation
       * (arXiv preprint 2022) **NUWA-Infinity: Autoregressive over Autoregressive Generation for Infinite Visual Synthesis**, Chenfei Wu et al. [[Paper](https://arxiv.org/abs/2207.09814)] [[Code](https://github.com/microsoft/NUWA)] [[Project](https://nuwa-infinity.microsoft.com/#/)]
         * 📚Unconditional Image Generation(HD), Text-to-Image(HD), Image Animation(HD), Image Outpainting(HD), Text-to-Video(HD)
       * (ECCV 2022) **NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion**, Chenfei Wu et al. [[Paper](https://arxiv.org/abs/2111.12417)] [[Code](https://github.com/microsoft/NUWA)]
         * 📚Text-To-Image, Sketch-to-Image, Image Completion, Text-Guided Image Manipulation, Text-to-Video, Video Prediction, Sketch-to-Video, Text-Guided Video Manipulation
       * (ACMMM 2022) **Rethinking Super-Resolution as Text-Guided Details Generation**, Chenxi Ma et al. [[Paper](https://arxiv.org/abs/2207.06604)] 
         * 📚Text-to-Image, High-resolution, Text-guided High-resolution
       * (arXiv preprint 2022) **Discrete Contrastive Diffusion for Cross-Modal and Conditional Generation**, Ye Zhu et al. [[Paper](https://arxiv.org/abs/2206.07771)] [[Code](https://github.com/L-YeZhu/CDCD)] 
         * 📚Text-to-Image, Dance-to-Music, Class-to-Image
       * (arXiv preprint 2022) **M6-Fashion: High-Fidelity Multi-modal Image Generation and Editing**, Zhikang Li et al. [[Paper](https://arxiv.org/abs/2205.11705)] 
         * 📚Text-to-Image, Unconditional Image Generation, Local-editing, Text-guided Local-editing, In/Out-painting, Style-mixing
       * (CVPR 2022) **Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning**, Yogesh Balaji et al. [[Paper](https://arxiv.org/abs/2203.02573)] [[Code](https://github.com/snap-research/MMVID)] [Project](https://snap-research.github.io/MMVID/)
         * 📚Text-to-Video, Independent Multimodal Controls, Dependent Multimodal Controls
       * ⭐⭐(CVPR 2022) **High-Resolution Image Synthesis with Latent Diffusion Models**, Robin Rombach et al. [[Paper](https://arxiv.org/abs/2112.10752)] [[Code](https://github.com/CompVis/latent-diffusion)] [[Stable Diffusion Code](https://github.com/CompVis/stable-diffusion)]
         * 📚Text-to-Image, Conditional Latent Diffusion, Super-Resolution, Inpainting
       * ⭐⭐(arXiv preprint 2022) **Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework**, Peng Wang et al. [[Paper](https://arxiv.org/abs/2202.03052v1)]  [[Code](https://github.com/ofa-sys/ofa)] [[Hugging Face](https://huggingface.co/OFA-Sys)]
         * 📚Text-to-Image Generation, Image Captioning, Text Summarization, Self-Supervised Image Classification, **[SOTA]** Referring Expression Comprehension, Visual Entailment, Visual Question Answering
       * (NeurIPS 2021) **M6-UFC: Unifying Multi-Modal Controls for Conditional Image Synthesis via Non-Autoregressive Generative Transformers**, Zhu Zhang et al. [[Paper](https://arxiv.org/abs/2105.14211)] 
         * 📚Text-to-Image, Sketch-to-Image, Style Transfer, Image Inpainting, Multi-Modal Control to Image
       * (arXiv preprint 2021) **ERNIE-ViLG: Unified Generative Pre-training for Bidirectional Vision-Language Generation**, Han Zhang et al. [[Paper](https://arxiv.org/abs/2112.15283)] 
         * A pre-trained **10-billion** parameter model: ERNIE-ViLG.
         * A large-scale dataset of **145 million** high-quality Chinese image-text pairs.
         * 📚Text-to-Image, Image Captioning,  Generative Visual Question Answering
       * (arXiv preprint 2021) **Multimodal Conditional Image Synthesis with Product-of-Experts GANs**, Xun Huang et al. [[Paper](https://arxiv.org/abs/2112.05130)]  [[Project](https://deepimagination.cc/PoE-GAN/)]
         * 📚Text-to-Image, Segmentation-to-Image, Text+Segmentation/Sketch/Image → Image, Sketch+Segmentation/Image → Image, Segmentation+Image → Image
       * (arXiv preprint 2021) **L-Verse: Bidirectional Generation Between Image and Text**, Taehoon Kim et al. [[Paper](https://arxiv.org/abs/2111.11133)] [[Code](https://github.com/tgisaturday/L-Verse)] 
         * 📚Text-To-Image, Image-To-Text, Image Reconstruction 
       * (arXiv preprint 2021) [💬Semantic Diffusion Guidance] **More Control for Free! Image Synthesis with Semantic Diffusion Guidance**, Xihui Liu et al. [[Paper](https://arxiv.org/abs/2112.05744)] [[Project](https://xh-liu.github.io/sdg/)] 
         * 📚Text-To-Image, Image-To-Image, Text+Image → Image 


   * <span id="head-APP"> **🛫Applications🛫** </span> **[       «🎯Back To Top»       ](#)**
       * (arXiv preprint 2023) [💬Grounded Generation] **Guiding Text-to-Image Diffusion Model Towards Grounded Generation**, Ziyi Li et al. [[Paper](https://arxiv.org/abs/2301.05221)] [[Code](https://github.com/Lipurple/Grounded-Diffusion)] [Project](https://lipurple.github.io/Grounded_Diffusion/)]
       * (arXiv preprint 2022) [💬Semantic segmentation] **CLIP is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation**, Yuqi Lin et al. [[Paper](https://arxiv.org/abs/2212.09506)] [[Code](https://github.com/linyq2117/CLIP-ES)]
       * (arXiv preprint 2022) [💬Unsupervised semantic segmentation] **Peekaboo: Text to Image Diffusion Models are Zero-Shot Segmentors**, Ryan Burgert et al. [[Paper](https://arxiv.org/abs/2211.13224)]
       * (SIGGRAPH Asia 2022) [💬Text+Speech → Gesture] **Rhythmic Gesticulator: Rhythm-Aware Co-Speech Gesture Synthesis with Hierarchical Neural Embeddings**, Tenglong Ao et al. [[Paper](https://arxiv.org/abs/2210.01448)] [[Code](https://github.com/Aubrey-ao/HumanBehaviorAnimation)]
       * (arXiv preprint 2022) [💬Text+Image+Shape → Image] **Shape-Guided Diffusion with Inside-Outside Attention**, Dong Huk Park et al. [[Paper](https://arxiv.org/abs/2212.00210v1)] [[Project](https://shape-guided-diffusion.github.io/)]

   * <span id="head-TI2I"> **Text+Image/Video → Image/Video** </span> **[       «🎯Back To Top»       ](#)**
       * (arXiv preprint 2023) [💬Fashion Image Editing] **FICE: Text-Conditioned Fashion Image Editing With Guided GAN Inversion**, Martin Pernuš et al. [[Paper](https://arxiv.org/abs/2301.02110)] [[Code](https://github.com/MartinPernus/FICE)] 
       * (AAAI 2023) **CLIPVG: Text-Guided Image Manipulation Using Differentiable Vector Graphics**, Yiren Song et al. [[Paper](https://arxiv.org/abs/2212.02122v1)] 
       * (AAAI 2023) **DE-Net: Dynamic Text-guided Image Editing Adversarial Networks**, Ming Tao et al. [[Paper](https://arxiv.org/abs/2206.01160)] [[Code](https://github.com/tobran/DE-Net)]
       * (arXiv preprint 2022) **Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation**, Narek Tumanyan et al. [[Paper](https://arxiv.org/abs/2211.12572)] [[Project](https://pnp-diffusion.github.io/)]
       * (arXiv preprint 2022) [💬Text+Image → Video] **Tell Me What Happened: Unifying Text-guided Video Completion via Multimodal Masked Video Generation**, Tsu-Jui Fu et al. [[Paper](https://arxiv.org/abs/2211.12824)]
       * (arXiv preprint 2022) **Null-text Inversion for Editing Real Images using Guided Diffusion Models**, Ron Mokady et al. [[Paper](https://arxiv.org/abs/2211.09794)] [[Project]([https://www.timothybrooks.com/instruct-pix2pix](https://null-text-inversion.github.io/))] 
       * (arXiv preprint 2022) **InstructPix2Pix: Learning to Follow Image Editing Instructions**, Tim Brooks et al. [[Paper](https://arxiv.org/abs/2211.09800)] [[Project](https://www.timothybrooks.com/instruct-pix2pix)] 
       * (ECCV 2022) [💬Style Transfer] **Language-Driven Artistic Style Transfer**, Tsu-Jui Fu et al. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-20059-5_41)] [[Code](https://github.com/tsujuifu/pytorch_ldast)]
       * (arXiv preprint 2022) **Bridging CLIP and StyleGAN through Latent Alignment for Image Editing**, Wanfeng Zheng et al. [[Paper](https://arxiv.org/abs/2210.04506)] 
       * (arXiv preprint 2022) **DiffEdit: Diffusion-based semantic image editing with mask guidance**, Guillaume Couairon et al. [[Paper](https://arxiv.org/abs/2210.11427v1)] 
       * (NeurIPS 2022) **One Model to Edit Them All: Free-Form Text-Driven Image Manipulation with Semantic Modulations**, Yiming Zhu et al. [[Paper](https://arxiv.org/abs/2210.07883)] [[Code](https://github.com/KumapowerLIU/FFCLIP)]
       * (BMVC 2022) **LDEdit: Towards Generalized Text Guided Image Manipulation via Latent Diffusion Models**, Paramanand Chandramouli et al. [[Paper](https://arxiv.org/abs/2210.02249v1)]
       * (ACMMM 2022) [💬Iterative Language-based Image Manipulation] **LS-GAN: Iterative Language-based Image Manipulation via Long and Short Term Consistency Reasoning**, Gaoxiang Cong et al. [[Paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548206)] 
       * (ACMMM 2022) [💬Digital Art Synthesis] **Draw Your Art Dream: Diverse Digital Art Synthesis with Multimodal Guided Diffusion**, Huang Nisha et al. [[Paper](https://arxiv.org/abs/2209.13360)] [[Code](https://github.com/haha-lisa/MGAD-multimodal-guided-artwork-diffusion)]
       * (SIGGRAPH Asia 2022) [💬HDR Panorama Generation] **Text2Light: Zero-Shot Text-Driven HDR Panorama Generation**, Zhaoxi Chen et al. [[Paper](https://arxiv.org/abs/2209.09898)] [[Project](https://frozenburning.github.io/projects/text2light/)] [[Code](https://github.com/FrozenBurning/Text2Light)]
       * (arXiv preprint 2022) **LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data**, Jihye Park et al. [[Paper](https://arxiv.org/abs/2208.14889)] [[Project](https://ku-cvlab.github.io/LANIT/)] [[Code](https://github.com/KU-CVLAB/LANIT)]
       * (ACMMM PIES-ME 2022) [💬3D Semantic Style Transfer] **Language-guided Semantic Style Transfer of 3D Indoor Scenes**, Bu Jin et al. [[Paper](https://arxiv.org/abs/2208.07870)] [[Code](https://github.com/AIR-DISCOVER/LASST)]
       * (arXiv preprint 2022) **DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation**, Nataniel Ruiz et al. [[Paper](https://arxiv.org/abs/2208.12242)] [[Project](https://dreambooth.github.io/)]
       * (arXiv preprint 2022) [💬Face Animation] **Language-Guided Face Animation by Recurrent StyleGAN-based Generator**, Tiankai Hang et al. [[Paper](https://arxiv.org/abs/2208.05617)] [[Code](https://github.com/TiankaiHang/language-guided-animation)]
       * (arXiv preprint 2022) [💬Fashion Design] **ARMANI: Part-level Garment-Text Alignment for Unified Cross-Modal Fashion Design**, Xujie Zhang et al. [[Paper](https://arxiv.org/abs/2208.05621)] [[Code](https://github.com/Harvey594/ARMANI)]
       * (arXiv preprint 2022) [💬Image Colorization] **TIC: Text-Guided Image Colorization**, Subhankar Ghosh et al.  [[Paper](https://arxiv.org/abs/2208.02843)] 
       * (ECCV 2022) [💬Pose Synthesis] **TIPS: Text-Induced Pose Synthesis**, Prasun Roy et al. [[Paper](https://arxiv.org/abs/2207.11718)] [[Code](https://github.com/prasunroy/tips)] [[Project](https://prasunroy.github.io/tips/)]
       * (ACMMM 2022) [💬Person Re-identification] **Learning Granularity-Unified Representations for Text-to-Image Person Re-identification**, Zhiyin Shao et al. [[Paper](https://arxiv.org/abs/2207.07802)] [[Code](https://github.com/ZhiyinShao-H/LGUR)]
       * (ACMMM 2022) **Towards Counterfactual Image Manipulation via CLIP**, Yingchen Yu et al. [[Paper](https://arxiv.org/abs/2207.02812)] [[Code](https://github.com/yingchen001/CF-CLIP)]
       * (ACMMM 2022) [💬Monocular Depth Estimation] **Can Language Understand Depth?**, Wangbo Zhao et al. [[Paper](https://arxiv.org/abs/2207.01077)] [[Code](https://github.com/Adonis-galaxy/DepthCLIP)]
       * (arXiv preprint 2022) [💬Image Style Transfer] **Referring Image Matting**, Tsu-Jui Fu et al. [[Paper](https://arxiv.org/abs/2106.00178)]
       * (CVPR 2022) [💬Image Segmentation] **Image Segmentation Using Text and Image Prompts**, Timo Lüddecke et al. [[Paper](https://arxiv.org/abs/2112.10003)] [[Code](https://github.com/timojl/clipseg)] 
       * (CVPR 2022) [💬Video Segmentation] **Modeling Motion with Multi-Modal Features for Text-Based Video Segmentation**, Wangbo Zhao et al. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Modeling_Motion_With_Multi-Modal_Features_for_Text-Based_Video_Segmentation_CVPR_2022_paper.pdf)] [[Code](https://github.com/wangbo-zhao/2022cvpr-mmmmtbvs)]
       * (arXiv preprint 2022) [💬Image Matting] **Referring Image Matting**, Sebastian Loeschcke et al. [[Paper](https://arxiv.org/abs/2206.05149)] [[Dataset](https://github.com/JizhiziLi/RIM)]
       * (arXiv preprint 2022) [💬Stylizing Video Objects] **Text-Driven Stylization of Video Objects**, Sebastian Loeschcke et al. [[Paper](https://arxiv.org/abs/2206.12396)] [[Project](https://sloeschcke.github.io/Text-Driven-Stylization-of-Video-Objects/)]
       * (arXiv preprint 2022) **DALL-E for Detection: Language-driven Context Image Synthesis for Object Detection**, Yunhao Ge et al. [[Paper](https://arxiv.org/abs/2206.09592)] 
       * (arXiv preprint 2022) [💬Animating Human Meshes] **CLIP-Actor: Text-Driven Recommendation and Stylization for Animating Human Meshes**, Kim Youwang et al. [[Paper](https://arxiv.org/abs/2206.04382)] [[Code](https://github.com/Youwang-Kim/CLIP-Actor)]
       * (arXiv preprint 2022) **Blended Latent Diffusion**, Omri Avrahami et al. [[Paper](https://arxiv.org/abs/2206.02779)] [[Code](https://github.com/omriav/blended-latent-diffusion)] [[Project](https://omriavrahami.com/blended-latent-diffusion-page/)]
       * (IEEE Transactions on Neural Networks and Learning Systems 2022) [💬Pose-Guided Person Generation] **Verbal-Person Nets: Pose-Guided Multi-Granularity Language-to-Person Generation**, Deyin Liu et al. [[Paper](https://ieeexplore.ieee.org/document/9732175)]
       * (SIGGRAPH 2022) [💬3D Avatar Generation] **AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars**, Fangzhou Hong et al. [[Paper](https://arxiv.org/abs/2205.08535)] [[Code](https://github.com/hongfz16/AvatarCLIP)] [[Project](https://hongfz16.github.io/projects/AvatarCLIP.html)] 
       * ⭐⭐(arXiv preprint 2022) [💬Image & Video Editing] **Text2LIVE: Text-Driven Layered Image and Video Editing**, Omer Bar-Tal et al. [[Paper](https://arxiv.org/abs/2204.02491)] [[Project](https://text2live.github.io/)] 
       * (Machine Vision and Applications 2022) **Paired-D++ GAN for image manipulation with text**, Duc Minh Vo et al. [[Paper](https://link.springer.com/article/10.1007/s00138-022-01298-7)]
       * (CVPR 2022) [💬Hairstyle Transfer] **HairCLIP: Design Your Hair by Text and Reference Image**, Tianyi Wei et al. [[Paper](https://arxiv.org/abs/2112.05142)] [[Code](https://github.com/wty-ustc/HairCLIP)] 
       * (CVPR 2022) [💬NeRF] **CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields**, Can Wang et al. [[Paper](https://arxiv.org/abs/2112.05139)] [[Code](https://github.com/cassiePython/CLIPNeRF)] [[Project](https://cassiepython.github.io/clipnerf/)]
       * (CVPR 2022) **DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation**, Gwanghyun Kim et al. [[Paper](https://arxiv.org/abs/2110.02711)]
       * (CVPR 2022) **ManiTrans: Entity-Level Text-Guided Image Manipulation via Token-wise Semantic Alignment and Generation**, Jianan Wang et al. [[Paper](https://arxiv.org/abs/2204.04428)] [[Project](https://jawang19.github.io/manitrans/)] 
       * (CVPR 2022) **Blended Diffusion for Text-driven Editing of Natural Images**, Omri Avrahami et al. [[Paper](https://arxiv.org/abs/2111.14818)] [[Code](https://github.com/omriav/blended-diffusion)] [[Project](https://omriavrahami.com/blended-diffusion-page/)] 
       * (CVPR 2022) **Predict, Prevent, and Evaluate: Disentangled Text-Driven Image Manipulation Empowered by Pre-Trained Vision-Language Model**, Zipeng Xu et al. [[Paper](https://arxiv.org/abs/2111.13333)] [[Code](https://github.com/zipengxuc/PPE-Pytorch)] 
       * (CVPR 2022) [💬Style Transfer] **CLIPstyler: Image Style Transfer with a Single Text Condition**, Gihyun Kwon et al. [[Paper](https://arxiv.org/abs/2112.00374)] [[Code](https://github.com/paper11667/CLIPstyler)] 
       * (arXiv preprint 2022) [💬Multi-person Image Generation] **Pose Guided Multi-person Image Generation From Text**, Soon Yau Cheong et al. [[Paper](https://arxiv.org/abs/2203.04907)]
       * (arXiv preprint 2022) [💬Image Style Transfer] **StyleCLIPDraw: Coupling Content and Style in Text-to-Drawing Translation**, Peter Schaldenbrand et al. [[Paper](https://arxiv.org/abs/2202.12362)] [[Dataset](https://www.kaggle.com/pittsburghskeet/drawings-with-style-evaluation-styleclipdraw)] [[Code](https://github.com/pschaldenbrand/StyleCLIPDraw)] [[Demo](https://replicate.com/pschaldenbrand/style-clip-draw)]
       * (arXiv preprint 2022) [💬Image Style Transfer] **Name Your Style: An Arbitrary Artist-aware Image Style Transfer**, Zhi-Song Liu et al. [[Paper](https://arxiv.org/abs/2202.13562)]
       * (arXiv preprint 2022) [💬3D Avatar Generation] **Text and Image Guided 3D Avatar Generation and Manipulation**, Zehranaz Canfes et al. [[Paper](https://arxiv.org/abs/2202.06079)] [[Project](https://catlab-team.github.io/latent3D/)]
       * (arXiv preprint 2022) [💬Image Inpainting] **NÜWA-LIP: Language Guided Image Inpainting with Defect-free VQGAN**, Minheng Ni et al. [[Paper](https://arxiv.org/abs/2202.05009)]
       * ⭐(arXiv preprint 2021) [💬Text+Image → Video] **Make It Move: Controllable Image-to-Video Generation with Text Descriptions**, Yaosi Hu et al. [[Paper](https://arxiv.org/abs/2112.02815)]
       * (arXiv preprint 2021) [💬NeRF] **Zero-Shot Text-Guided Object Generation with Dream Fields**, Ajay Jain et al. [[Paper](https://arxiv.org/abs/2112.01455)]  [[Project](https://ajayj.com/dreamfields)]
       * (NeurIPS 2021) **Instance-Conditioned GAN**, Arantxa Casanova et al. [[Paper](https://arxiv.org/abs/2109.05070)] [[Code](https://github.com/facebookresearch/ic_gan)]
       * (ICCV 2021) **Language-Guided Global Image Editing via Cross-Modal Cyclic Mechanism**, Wentao Jiang et al. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Language-Guided_Global_Image_Editing_via_Cross-Modal_Cyclic_Mechanism_ICCV_2021_paper.pdf)]
       * (ICCV 2021) **Talk-to-Edit: Fine-Grained Facial Editing via Dialog**, Yuming Jiang et al. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Talk-To-Edit_Fine-Grained_Facial_Editing_via_Dialog_ICCV_2021_paper.pdf)] [[Project](https://www.mmlab-ntu.com/project/talkedit/)] [[Code](https://github.com/yumingj/Talk-to-Edit)]
       * (ICCVW 2021) **CIGLI: Conditional Image Generation from Language & Image**, Xiaopeng Lu et al. [[Paper](https://openaccess.thecvf.com/content/ICCV2021W/CLVL/papers/Lu_CIGLI_Conditional_Image_Generation_From_Language__Image_ICCVW_2021_paper.pdf)] [[Code](https://github.com/vincentlux/CIGLI?utm_source=catalyzex.com)]
       * (ICCV 2021) **StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery**, Or Patashnik et al. [[Paper](https://arxiv.org/abs/2103.17249)] [[Code](https://github.com/orpatashnik/StyleCLIP)]
       * (arXiv preprint 2021) **Paint by Word**, David Bau et al. [[Paper](https://arxiv.org/pdf/2103.10951.pdf)] 
       * ⭐(arXiv preprint 2021) **Zero-Shot Text-to-Image Generation**, Aditya Ramesh et al. [[Paper](https://arxiv.org/pdf/2102.12092.pdf)] [[Code](https://github.com/openai/DALL-E)] [[Blog](https://openai.com/blog/dall-e/)] [[Model Card](https://github.com/openai/DALL-E/blob/master/model_card.md)] [[Colab](https://colab.research.google.com/drive/1KA2w8bA9Q1HDiZf5Ow_VNOrTaWW4lXXG?usp=sharing)] 
       * (NeurIPS 2020) **Lightweight Generative Adversarial Networks for Text-Guided Image Manipulation**, Bowen Li et al. [[Paper](https://arxiv.org/pdf/2010.12136.pdf)]
       * (CVPR 2020) **ManiGAN: Text-Guided Image Manipulation**, Bowen Li et al. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_ManiGAN_Text-Guided_Image_Manipulation_CVPR_2020_paper.pdf)] [[Code](https://github.com/mrlibw/ManiGAN)]
       * (ACMMM 2020) **Text-Guided Neural Image Inpainting**, Lisai Zhang et al. [[Paper](https://arxiv.org/pdf/2004.03212.pdf)] [[Code](https://github.com/idealwhite/TDANet)]
       * (ACMMM 2020) **Describe What to Change: A Text-guided Unsupervised Image-to-Image Translation Approach**, Yahui Liu et al. [[Paper](https://arxiv.org/pdf/2008.04200.pdf)]
       * (NeurIPS 2018) **Text-adaptive generative adversarial networks: Manipulating images with natural language**, Seonghyeon Nam et al. [[Paper](http://papers.nips.cc/paper/7290-text-adaptive-generative-adversarial-networks-manipulating-images-with-natural-language.pdf)] [[Code](https://github.com/woozzu/tagan)]

   * <span id="head-ATI2I"> **Audio+Text+Image/Video → Image/Video** </span> **[       «🎯Back To Top»       ](#)**
       * (arXiv preprint 2022) **Robust Sound-Guided Image Manipulation**, Seung Hyun Lee et al. [[Paper](https://arxiv.org/abs/2208.14114)]
       
   * <span id="head-L2I"> **Layout → Image** </span> **[       «🎯Back To Top»       ](#)**
       * (CVPR 2022) **Modeling Image Composition for Complex Scene Generation**, Zuopeng Yang et al. [[Paper](https://arxiv.org/abs/2206.00923)] [[Code](https://github.com/JohnDreamer/TwFA)]
       * (CVPR 2022) **Interactive Image Synthesis with Panoptic Layout Generation**, Bo Wang et al. [[Paper](https://arxiv.org/abs/2203.02104)] 
       * (CVPR 2021 [AI for Content Creation Workshop](http://visual.cs.brown.edu/workshops/aicc2021/)) **High-Resolution Complex Scene Synthesis with Transformers**, Manuel Jahn et al. [[Paper](https://arxiv.org/pdf/2105.06458.pdf)] 
       * (CVPR 2021) **Context-Aware Layout to Image Generation with Enhanced Object Appearance**, Sen He et al. [[Paper](https://arxiv.org/pdf/2103.11897.pdf)] [[Code](https://github.com/wtliao/layout2img)] 

   * <span id="head-L2S"> **Label-set → Semantic maps** </span> **[       «🎯Back To Top»       ](#)**
       * (ECCV 2020) **Controllable image synthesis via SegVAE**, Yen-Chi Cheng et al. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520154.pdf)] [[Code](https://github.com/yccyenchicheng/SegVAE)]
       
   * <span id="head-S2I"> **Speech → Image** </span> **[       «🎯Back To Top»       ](#)**
       *  (IEEE/ACM Transactions on Audio, Speech and Language Processing-2021) **Generating Images From Spoken Descriptions**, Xinsheng Wang et al. [[Paper](https://dl.acm.org/doi/10.1109/TASLP.2021.3053391)] [[Code](https://github.com/xinshengwang/S2IGAN)]  [[Project](https://xinshengwang.github.io/project/s2igan/)]
       *  (INTERSPEECH 2020)**[Extent Version👆] S2IGAN: Speech-to-Image Generation via Adversarial Learning**, Xinsheng Wang et al. [[Paper](https://arxiv.org/abs/2005.06968)]
       * (IEEE Journal of Selected Topics in Signal Processing-2020) **Direct Speech-to-Image Translation**, Jiguo Li et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9067083)] [[Code](https://github.com/smallflyingpig/speech-to-image-translation-without-text)] [[Project](https://smallflyingpig.github.io/speech-to-image/main)]
       
   * <span id="head-T2VR"> **Text → Visual Retrieval** </span> **[       «🎯Back To Top»       ](#)**
       * (ACMMM 2022) **CAIBC: Capturing All-round Information Beyond Color for Text-based Person Retrieval**, Zijie Wang et al. [[Paper](https://arxiv.org/abs/2209.05773)] 
       * (AAAI 2022) **Cross-Modal Coherence for Text-to-Image Retrieval**, Malihe Alikhani et al. [[Paper](https://arxiv.org/abs/2109.11047)] 
       * (ECCV [RWS 2022](https://vap.aau.dk/rws-eccv2022/)) [💬Person Retrieval] **See Finer, See More: Implicit Modality Alignment for Text-based Person Retrieval**, Xiujun Shu et al. [[Paper](https://arxiv.org/abs/2208.08608)] [[Code](https://github.com/TencentYoutuResearch/PersonRetrieval-IVT)] 
       * (ECCV 2022) [💬Text+Sketch→Visual Retrieval] **A Sketch Is Worth a Thousand Words: Image Retrieval with Text and Sketch**, Patsorn Sangkloy et al. [[Paper](https://arxiv.org/abs/2208.03354)] [[Project](https://patsorn.me/projects/tsbir/)] 
       * (Neurocomputing-2022) **TIPCB: A simple but effective part-based convolutional baseline for text-based person search**, Yuhao Chen et al. [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231222004726)] [[Code](https://github.com/OrangeYHChen/TIPCB?utm_source=catalyzex.com)] 
       * (arXiv preprint 2021) [💬Dataset] **FooDI-ML: a large multi-language dataset of food, drinks and groceries images and descriptions**, David Amat Olóndriz et al. [[Paper](https://arxiv.org/abs/2110.02035)] [[Code](https://github.com/glovo/foodi-ml-dataset)] 
       * (CVPRW 2021) **TIED: A Cycle Consistent Encoder-Decoder Model for Text-to-Image Retrieval**, Clint Sebastian et al. [[Paper](https://openaccess.thecvf.com/content/CVPR2021W/AICity/papers/Sebastian_TIED_A_Cycle_Consistent_Encoder-Decoder_Model_for_Text-to-Image_Retrieval_CVPRW_2021_paper.pdf)] 
       * (CVPR 2021) **T2VLAD: Global-Local Sequence Alignment for Text-Video Retrieval**, Xiaohan Wang et al. [[Paper](https://arxiv.org/pdf/2104.10054.pdf)] 
       * (CVPR 2021) **Thinking Fast and Slow: Efficient Text-to-Visual Retrieval with Transformers**, Antoine Miech et al. [[Paper](https://arxiv.org/pdf/2103.16553.pdf)] 
       * (IEEE Access 2019) **Query is GAN: Scene Retrieval With Attentional Text-to-Image Generative Adversarial Network**, RINTARO YANAGI et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8868179)]
 
   * <span id="head-T2M"> **Text → 3D/Motion/Shape/Mesh/Object...** </span> **[       «🎯Back To Top»       ](#)**
       * (arXiv preprint 2022) [💬Text → 3D] **Dream3D: Zero-Shot Text-to-3D Synthesis Using 3D Shape Prior and Text-to-Image Diffusion Models**, Jiale Xu et al.  [[Paper](https://arxiv.org/abs/2212.14704)] [[Project](https://bluestyle97.github.io/dream3d/)]
       * (arXiv preprint 2022) [💬3D Generative Model] **DATID-3D: Diversity-Preserved Domain Adaptation Using Text-to-Image Diffusion for 3D Generative Model**, Gwanghyun Kim et al. [[Paper](https://arxiv.org/abs/2211.16374)] [[Code](https://github.com/gwang-kim/DATID-3D)] [[Project](https://datid-3d.github.io/)]
       * (arXiv preprint 2022) [💬Point Clouds] **Point-E: A System for Generating 3D Point Clouds from Complex Prompts**, Alex Nichol et al.  [[Paper](https://arxiv.org/abs/2212.08751)] [[Code](https://github.com/openai/point-e)]
       * (arXiv preprint 2022) [💬Text → 3D] **Magic3D: High-Resolution Text-to-3D Content Creation**, Chen-Hsuan Lin et al.  [[Paper](https://arxiv.org/abs/2211.10440)] [[Project](https://deepimagination.cc/Magic3D/)]
       * (arXiv preprint 2022) [💬Text → Shape] **Diffusion-SDF: Text-to-Shape via Voxelized Diffusion**, Muheng Li et al. [[Paper](https://arxiv.org/abs/2212.03293)] [[Code](https://github.com/ttlmh/Diffusion-SDF)]
       * (NIPS 2022) [💬Mesh] **TANGO: Text-driven Photorealistic and Robust 3D Stylization via Lighting Decomposition**, Yongwei Chen et al. [[Paper](https://arxiv.org/abs/2210.11277)] [[Project](https://cyw-3d.github.io/tango/)] [[Code](https://github.com/Gorilla-Lab-SCUT/tango)] 
       * (arXiv preprint 2022) [💬Human Motion Generation] **Human Motion Diffusion Model**, Guy Tevet et al. [[Paper](https://arxiv.org/abs/2209.14916)] [[Project](https://guytevet.github.io/mdm-page/)] [[Code](https://github.com/GuyTevet/motion-diffusion-model)]
       * (arXiv preprint 2022) [💬Human Motion Generation] **MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model**, Mingyuan Zhang et al. [[Paper](https://arxiv.org/abs/2208.15001)] [[Project](https://mingyuan-zhang.github.io/projects/MotionDiffuse.html#)]
       * (arXiv preprint 2022) [💬3D]**DreamFusion: Text-to-3D using 2D Diffusion**, Ben Poole et al.  [[Paper](https://arxiv.org/abs/2209.14988)] [[Project](https://dreamfusion3d.github.io/)] [[Short Read](https://www.louisbouchard.ai/dreamfusion/)]
       * (arXiv preprint 2022) [💬3D Shape] **ISS: Image as Stetting Stone for Text-Guided 3D Shape Generation**, Zhengzhe Liu et al. [[Paper](https://arxiv.org/abs/2209.04145)]
       * (ECCV 2022) [💬Virtual Humans] **Compositional Human-Scene Interaction Synthesis with Semantic Control**, Kaifeng Zhao et al. [[Paper](https://arxiv.org/abs/2207.12824)] [[Project](https://zkf1997.github.io/COINS/index.html)] [[Code](https://github.com/zkf1997/COINS)] 
       * (CVPR 2022) [💬3D Shape] **Towards Implicit Text-Guided 3D Shape Generation**, Zhengzhe Liu et al. [[Paper](https://arxiv.org/abs/2203.14622)] [[Code](https://github.com/liuzhengzhe/Towards-Implicit-Text-Guided-Shape-Generation)]
       * (CVPR 2022) [💬Object] **Zero-Shot Text-Guided Object Generation with Dream Fields**, Ajay Jain et al. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Jain_Zero-Shot_Text-Guided_Object_Generation_With_Dream_Fields_CVPR_2022_paper.pdf)] [[Project](https://ajayj.com/dreamfields)] [[Code](https://github.com/google-research/google-research/tree/master/dreamfields)] 
       * (CVPR 2022) [💬Mesh] **Text2Mesh: Text-Driven Neural Stylization for Meshes**, Oscar Michel et al. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Michel_Text2Mesh_Text-Driven_Neural_Stylization_for_Meshes_CVPR_2022_paper.pdf)] [[Project](https://threedle.github.io/text2mesh/)] [[Code](https://github.com/threedle/text2mesh)] 
       * (CVPR 2022) [💬Motion] **Generating Diverse and Natural 3D Human Motions from Text**, Chuan Guo et al. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.pdf)] [[Project](https://ericguo5513.github.io/text-to-motion/)] [[Code](https://github.com/EricGuo5513/text-to-motion)] 
       * (CVPR 2022) [💬Shape] **CLIP-Forge: Towards Zero-Shot Text-to-Shape Generation**, Aditya Sanghi et al. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Sanghi_CLIP-Forge_Towards_Zero-Shot_Text-To-Shape_Generation_CVPR_2022_paper.pdf)] [[Code](https://github.com/AutodeskAILab/Clip-Forge)] 
       * (arXiv preprint 2022) [💬Motion] **TEMOS: Generating diverse human motions from textual descriptions**, Mathis Petrovich et al. [[Paper](https://arxiv.org/abs/2204.14109)] [[Project](https://mathis.petrovich.fr/temos/)] [[Code](https://github.com/Mathux/TEMOS)] 
   
   * <span id="head-T2V"> **Text → Video** </span> **[       «🎯Back To Top»       ](#)**
       * ⭐(arXiv preprint 2023) **Text-To-4D Dynamic Scene Generation**, Uriel Singer et al. [[Paper](https://arxiv.org/abs/2301.11280)] [[Project](https://make-a-video3d.github.io/)]
       * (arXiv preprint 2022) **Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation**, Jay Zhangjie Wu et al. [[Paper](https://arxiv.org/abs/2212.11565)] [[Project](https://tuneavideo.github.io/)] [[Code](https://github.com/showlab/Tune-A-Video)]
       * (arXiv preprint 2022) **MagicVideo: Efficient Video Generation With Latent Diffusion Models**, Daquan Zhou et al. [[Paper](https://arxiv.org/abs/2211.11018)] [[Project](https://magicvideo.github.io/#)] 
       * (arXiv preprint 2022) **Imagen Video: High Definition Video Generation with Diffusion Models**, Jonathan Ho et al. [[Paper](https://arxiv.org/abs/2210.02303v1)] [[Project](https://imagen.research.google/video/)] 
       * (arXiv preprint 2022) **Text-driven Video Prediction**, Xue Song et al. [[Paper](https://arxiv.org/abs/2210.02872)] 
       * (arXiv preprint 2022) **Make-A-Video: Text-to-Video Generation without Text-Video Data**, Uriel Singer et al. [[Paper](https://arxiv.org/abs/2209.14792)] [[Project](https://makeavideo.studio/)] [[Short read](https://www.louisbouchard.ai/make-a-video/)] [[Code](https://github.com/lucidrains/make-a-video-pytorch)]
       * (ECCV 2022) [💬Story Continuation] **StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation**, Adyasha Maharana et al. [[Paper](https://arxiv.org/abs/2209.06192)] [[Code](https://github.com/adymaharana/storydalle)]
       * (arXiv preprint 2022) [💬Story → Video] **Word-Level Fine-Grained Story Visualization**, Bowen Li et al. [[Paper](https://arxiv.org/abs/2208.02341)] [[Code](https://github.com/mrlibw/Word-Level-Story-Visualization)]
       * (arXiv preprint 2022) **CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers**, Wenyi Hong et al. [[Paper](https://arxiv.org/abs/2205.15868)] [[Code](https://github.com/THUDM/CogVideo)]
       * (CVPR 2022) **Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning**, Yogesh Balaji et al. [[Paper](https://arxiv.org/abs/2203.02573)] [[Code](https://github.com/snap-research/MMVID)] [Project](https://snap-research.github.io/MMVID/)
       * (arXiv preprint 2022) **Video Diffusion Models**, Jonathan Ho et al. [[Paper](https://arxiv.org/abs/2204.03458)] [[Project](https://video-diffusion.github.io/)]
       * (arXiv preprint 2021) [❌Genertation Task] **Transcript to Video: Efficient Clip Sequencing from Texts**, Ligong Han et al. [[Paper](https://arxiv.org/pdf/2107.11851.pdf)] [[Project](http://www.xiongyu.me/projects/transcript2video/)]
       * (arXiv preprint 2021) **GODIVA: Generating Open-DomaIn Videos from nAtural Descriptions**, Chenfei Wu et al. [[Paper](https://arxiv.org/pdf/2104.14806.pdf)] 
       * (arXiv preprint 2021) **Text2Video: Text-driven Talking-head Video Synthesis with Phonetic Dictionary**, Sibo Zhang et al. [[Paper](https://arxiv.org/pdf/2104.14631.pdf)] 
       * (IEEE Access 2020) **TiVGAN: Text to Image to Video Generation With Step-by-Step Evolutionary Generator**, DOYEON KIM et al. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9171240)] 
       * (IJCAI 2019) **Conditional GAN with Discriminative Filter Generation for Text-to-Video Synthesis**, Yogesh Balaji et al. [[Paper](https://www.ijcai.org/Proceedings/2019/0276.pdf)] [[Code](https://github.com/minrq/CGAN_Text2Video)] 
       * (IJCAI 2019) **IRC-GAN: Introspective Recurrent Convolutional GAN for Text-to-video Generation**, Kangle Deng et al. [[Paper](https://www.ijcai.org/Proceedings/2019/0307.pdf)] 
       * (CVPR 2019) [💬Story → Video] **StoryGAN: A Sequential Conditional GAN for Story Visualization**, Yitong Li et al. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/12233https://openaccess.thecvf.com/content_CVPR_2019/html/Li_StoryGAN_A_Sequential_Conditional_GAN_for_Story_Visualization_CVPR_2019_paper.html)] [[Code](https://github.com/yitong91/StoryGAN?utm_source=catalyzex.com)]
       * (AAAI 2018) **Video Generation From Text**, Yitong Li et al. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/12233)] 
       * (ACMMM 2017) **To create what you tell: Generating videos from captions**, Yingwei Pan et al. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3123266.3127905)] 

   * <span id="head-T2Mu"> **Text → Music** </span> **[       «🎯Back To Top»       ](#)**
       * ⭐(arXiv preprint 2023) **MusicLM: Generating Music From Text**, Andrea Agostinelli et al. [[Paper](https://arxiv.org/abs/2301.11325)] [[Project](https://google-research.github.io/seanet/musiclm/examples/)] [[MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps)]


## <span id="head6"> *Contact Me* </span>

* [Yutong ZHOU](https://github.com/Yutong-Zhou-cv) in [Interaction Laboratory, Ritsumeikan University.](https://github.com/Rits-Interaction-Laboratory) ლ(╹◡╹ლ) 

* If you have any question, please feel free to contact Yutong ZHOU (E-mail: <zhou@i.ci.ritsumei.ac.jp>).

<!--
<a href='https://clustrmaps.com/site/1bpq4'  title='Visit tracker'><img src='//clustrmaps.com/map_v2.png?cl=ffffff&w=300&t=tt&d=3wemEIurL5hHYA1MNxyuQaTay2TnGf9HrAqmu53SPco'/></a>
-->
